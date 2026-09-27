from uuid import uuid4

from .audit import AuditTrail
from .domain import ConflictError, NotFoundError
from .rules import RuleEngine


# Candidate actions that cascade to the candidate's open observations.
_CASCADE_ACTIONS = {"withdraw", "reclassify"}


class DomainService:
    def __init__(self, repository, rules=None):
        self.repository = repository
        self.rules = rules or RuleEngine()
        self.audit = AuditTrail(repository)

    def _lookup(self, kind, field, value):
        return self.repository.find_entities(self.rules.normalize_kind(kind), field, value)

    def health(self):
        return {"status": "ok" if self.repository.ping() else "error"}

    def create(self, actor, kind, data, idempotency_key=None):
        kind = self.rules.normalize_kind(kind)
        payload = dict(data or {})
        if idempotency_key:
            existing = self.repository.get_idempotency(actor.user_id, idempotency_key)
            if existing:
                entity = self.repository.get_entity(existing)
                if entity:
                    return entity
        validated = self.rules.validate_create(actor, kind, payload, self._lookup)
        if validated:
            payload.update(validated)
        entity_id = str(payload.pop("id", "") or uuid4())
        if self.repository.get_entity(entity_id):
            raise ConflictError("entity already exists: " + entity_id)
        status = self.rules.initial_status(kind)
        entity = self.repository.create_entity(entity_id, kind, status, payload, actor.user_id)
        self.audit.record(entity_id, actor, "create", None, status, {"kind": kind})
        if idempotency_key:
            self.repository.save_idempotency(actor.user_id, idempotency_key, entity_id)
        return entity

    def transition(self, actor, entity_id, action, data=None, expected_version=None):
        entity = self.repository.get_entity(entity_id)
        if not entity:
            raise NotFoundError("entity not found: " + entity_id)
        expected = int(expected_version) if expected_version is not None else entity["version"]
        payload = dict(data or {})
        next_status, patch = self.rules.validate_transition(
            actor, entity, action, payload, self._lookup
        )
        kind = self.rules.normalize_kind(entity["kind"])
        if kind == "candidate" and action in _CASCADE_ACTIONS:
            return self._candidate_cascade(
                actor, entity, action, payload, patch, next_status, expected
            )
        merged = dict(entity["data"])
        merged.update(patch)
        updated = self.repository.update_entity(entity_id, expected, next_status, merged)
        self.audit.record(
            entity_id,
            actor,
            action,
            entity["status"],
            updated["status"],
            {"patch": patch},
        )
        return updated

    def _candidate_cascade(self, actor, candidate, action, payload, patch, next_status, expected):
        """Withdraw/reclassify a candidate together with its open observations."""
        candidate_data = dict(candidate["data"])
        candidate_data.update(patch)
        observations = self.repository.find_entities(
            "observation", "candidate_id", candidate["id"]
        )
        updates = []
        audits = []
        affected_ids = []
        if action == "withdraw":
            for observation in observations:
                obs_update, obs_audit, _ = self._observation_withdraw(
                    observation, actor, payload.get("reason"), candidate
                )
                if obs_update is None:
                    continue
                updates.append(obs_update)
                audits.append(obs_audit)
                affected_ids.append(observation["id"])
        else:  # reclassify
            for observation in observations:
                if observation["status"] != "scheduled":
                    continue
                obs_data = dict(observation["data"])
                obs_data["review_status"] = "pending"
                updates.append(
                    {
                        "id": observation["id"],
                        "expected_version": observation["version"],
                        "status": observation["status"],
                        "data": obs_data,
                    }
                )
                audits.append(
                    {
                        "entity_id": observation["id"],
                        "actor_id": actor.user_id,
                        "actor_role": actor.role,
                        "action": "flag_review",
                        "from_status": observation["status"],
                        "to_status": observation["status"],
                        "detail": {
                            "reason": "candidate_reclassified",
                            "candidate_id": candidate["id"],
                            "previous_type": patch.get("previous_type"),
                            "new_type": patch.get("transient_type"),
                        },
                    }
                )
                affected_ids.append(observation["id"])

        updates.append(
            {
                "id": candidate["id"],
                "expected_version": expected,
                "status": next_status,
                "data": candidate_data,
            }
        )
        audits.append(
            {
                "entity_id": candidate["id"],
                "actor_id": actor.user_id,
                "actor_role": actor.role,
                "action": action,
                "from_status": candidate["status"],
                "to_status": next_status,
                "detail": {"patch": patch},
            }
        )
        # The candidate is updated last inside the unit; a stale expected version
        # aborts everything before any row is committed.
        self.repository.apply_unit(updates, audits)
        updated_candidate = self.repository.get_entity(candidate["id"])
        updated_candidate["affected_observations"] = [
            self.repository.get_entity(entity_id) for entity_id in affected_ids
        ]
        return updated_candidate

    def _observation_withdraw(self, observation, actor, reason, candidate):
        """Build update/audit entries for one observation revoked by a candidate
        withdrawal. Completed or already withdrawn observations are untouched."""
        if observation["status"] not in ("requested", "scheduled"):
            return None, None, None
        _, obs_patch = self.rules.validate_transition(
            actor, observation, "withdraw", {"reason": reason}, self._lookup
        )
        obs_data = dict(observation["data"])
        obs_data.update(obs_patch)
        update = {
            "id": observation["id"],
            "expected_version": observation["version"],
            "status": "withdrawn",
            "data": obs_data,
        }
        audit = {
            "entity_id": observation["id"],
            "actor_id": actor.user_id,
            "actor_role": actor.role,
            "action": "withdraw",
            "from_status": observation["status"],
            "to_status": "withdrawn",
            "detail": {
                "patch": obs_patch,
                "cascade": "candidate_withdraw",
                "candidate_id": candidate["id"],
                "reason": reason,
            },
        }
        return update, audit, obs_patch

    def get(self, entity_id):
        entity = self.repository.get_entity(entity_id)
        if not entity:
            raise NotFoundError("entity not found: " + entity_id)
        return entity

    def list(self, kind=None, status=None):
        if kind:
            kind = self.rules.normalize_kind(kind)
        return self.repository.list_entities(kind=kind, status=status)

    def audit_log(self, entity_id=None):
        return self.repository.list_audit(entity_id=entity_id)
