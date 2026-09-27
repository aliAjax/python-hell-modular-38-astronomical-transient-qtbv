from uuid import uuid4

from .audit import AuditTrail
from .domain import ConflictError, NotFoundError
from .rules import RuleEngine


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
        if entity["kind"] == "candidate" and action == "withdraw":
            return self._withdraw_candidate(actor, entity, payload, expected)
        if entity["kind"] == "candidate" and action == "reclassify":
            return self._reclassify_candidate(actor, entity, payload, expected)
        next_status, patch = self.rules.validate_transition(
            actor, entity, action, payload, self._lookup
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

    def _candidate_observations(self, candidate_id, statuses):
        observations = self._lookup("observation", "candidate_id", candidate_id) or []
        return [obs for obs in observations if obs["status"] in statuses]

    def _withdraw_candidate(self, actor, entity, data, expected):
        next_status, patch = self.rules.validate_transition(
            actor, entity, "withdraw", data, self._lookup
        )
        reason = data.get("reason")
        pending = self._candidate_observations(entity["id"], ("requested", "scheduled"))
        merged = dict(entity["data"])
        merged.update(patch)
        updates = [(entity["id"], expected, next_status, merged)]
        for obs in pending:
            obs_data = dict(obs["data"])
            obs_data.update(
                {
                    "withdrawn_reason": reason,
                    "withdrawn_by": actor.user_id,
                    "withdrawn_with_candidate": entity["id"],
                }
            )
            updates.append((obs["id"], obs["version"], "withdrawn", obs_data))
        self.repository.update_entities(updates)
        self.audit.record(
            entity["id"], actor, "withdraw", entity["status"], next_status, {"patch": patch}
        )
        cancelled = []
        for obs in pending:
            self.audit.record(
                obs["id"],
                actor,
                "withdraw",
                obs["status"],
                "withdrawn",
                {
                    "reason": reason,
                    "candidate_id": entity["id"],
                    "cascade": "candidate_withdraw",
                },
            )
            cancelled.append(self.repository.get_entity(obs["id"]))
        updated = self.repository.get_entity(entity["id"])
        updated["cancelled_observations"] = cancelled
        return updated

    def _reclassify_candidate(self, actor, entity, data, expected):
        next_status, patch = self.rules.validate_transition(
            actor, entity, "reclassify", data, self._lookup
        )
        previous_type = entity["data"].get("transient_type")
        new_type = patch.get("transient_type")
        scheduled = self._candidate_observations(entity["id"], ("scheduled",))
        merged = dict(entity["data"])
        merged.update(patch)
        updates = [(entity["id"], expected, next_status, merged)]
        for obs in scheduled:
            obs_data = dict(obs["data"])
            obs_data.update(
                {
                    "review_pending": True,
                    "review_reason": "candidate reclassified from %s to %s"
                    % (previous_type, new_type),
                }
            )
            updates.append((obs["id"], obs["version"], obs["status"], obs_data))
        self.repository.update_entities(updates)
        self.audit.record(
            entity["id"], actor, "reclassify", entity["status"], next_status, {"patch": patch}
        )
        flagged = []
        for obs in scheduled:
            self.audit.record(
                obs["id"],
                actor,
                "flag_review",
                obs["status"],
                obs["status"],
                {
                    "previous_type": previous_type,
                    "new_type": new_type,
                    "candidate_id": entity["id"],
                },
            )
            flagged.append(self.repository.get_entity(obs["id"]))
        updated = self.repository.get_entity(entity["id"])
        updated["flagged_observations"] = flagged
        return updated

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
