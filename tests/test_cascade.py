import tempfile
import unittest
from pathlib import Path

from src.domain import Actor, ConflictError
from src.repository import SQLiteRepository
from src.rules import RuleEngine
from src.service import DomainService


class CascadeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = DomainService(
            SQLiteRepository(Path(self.tmp.name) / "cascade.db"),
            RuleEngine(),
        )
        self.analyst = Actor("analyst-1", "analyst")
        self.coordinator = Actor("coordinator-1", "coordinator")
        self.operator = Actor("operator-1", "operator")

    def tearDown(self):
        self.tmp.cleanup()

    def _setup_candidate_with_observations(self):
        source = self.service.create(
            self.analyst,
            "source",
            {"name": "Survey", "survey_name": "S"},
        )
        candidate = self.service.create(
            self.analyst,
            "candidate",
            {
                "source_id": source["id"],
                "event_id": "AT-CASCADE",
                "ra": 10,
                "dec": 20,
                "magnitude": 18,
                "transient_type": "unknown",
                "observed_at": "2026-09-27T00:00:00Z",
            },
        )
        telescope = self.service.create(
            self.coordinator,
            "telescope",
            {"name": "North 2m", "aperture_m": 2.0, "site_name": "NAO"},
        )
        window = {
            "start_at": "2026-09-28T10:00:00Z",
            "end_at": "2026-09-28T11:00:00Z",
        }
        later_window = {
            "start_at": "2026-09-28T12:00:00Z",
            "end_at": "2026-09-28T13:00:00Z",
        }

        def make_observation(obs_window):
            obs = self.service.create(
                self.coordinator,
                "observation",
                {
                    "candidate_id": candidate["id"],
                    "telescope_id": telescope["id"],
                    "team_id": "team-north",
                    "mode": "imaging",
                    **obs_window,
                },
            )
            return obs

        scheduled = make_observation(window)
        scheduled = self.service.transition(
            self.coordinator, scheduled["id"], "schedule", {"operator_id": "op-1"}
        )
        requested = make_observation(window)
        completed = make_observation(later_window)
        completed = self.service.transition(
            self.coordinator, completed["id"], "schedule", {"operator_id": "op-1"}
        )
        completed = self.service.transition(
            self.operator, completed["id"], "complete", {"quality": "good"}
        )
        return candidate, telescope, window, scheduled, requested, completed

    def test_withdraw_candidate_revokes_open_observations(self):
        candidate, telescope, window, scheduled, requested, completed = (
            self._setup_candidate_with_observations()
        )
        result = self.service.transition(
            self.analyst,
            candidate["id"],
            "withdraw",
            {"reason": "false positive after image subtraction"},
        )
        self.assertEqual(result["status"], "withdrawn")

        affected = {item["id"]: item for item in result["affected_observations"]}
        self.assertEqual(set(affected), {scheduled["id"], requested["id"]})
        for observation in affected.values():
            self.assertEqual(observation["status"], "withdrawn")
            self.assertEqual(
                observation["data"]["reason"], "false positive after image subtraction"
            )
            self.assertEqual(observation["data"]["withdrawn_by"], "analyst-1")
        # Completed observations are not undone and not listed.
        self.assertEqual(self.service.get(completed["id"])["status"], "completed")

        # Each cascade revocation is audited with reason and operator.
        obs_audit = [
            entry
            for entry in self.service.audit_log(scheduled["id"])
            if entry["action"] == "withdraw"
        ]
        self.assertEqual(len(obs_audit), 1)
        self.assertEqual(obs_audit[0]["actor_id"], "analyst-1")
        self.assertEqual(obs_audit[0]["from_status"], "scheduled")
        self.assertEqual(obs_audit[0]["to_status"], "withdrawn")
        self.assertEqual(
            obs_audit[0]["detail"]["reason"], "false positive after image subtraction"
        )
        self.assertEqual(obs_audit[0]["detail"]["cascade"], "candidate_withdraw")
        self.assertEqual(obs_audit[0]["detail"]["candidate_id"], candidate["id"])

    def test_withdraw_releases_telescope_and_team_windows(self):
        candidate, telescope, window, scheduled, requested, completed = (
            self._setup_candidate_with_observations()
        )
        self.service.transition(
            self.analyst, candidate["id"], "withdraw", {"reason": "false positive"}
        )

        # A fresh application can now claim the exact same telescope/team window.
        source = self.service.list("source")[0]
        other_candidate = self.service.create(
            self.analyst,
            "candidate",
            {
                "source_id": source["id"],
                "event_id": "AT-OTHER",
                "ra": 11,
                "dec": 21,
                "magnitude": 17,
                "transient_type": "unknown",
                "observed_at": "2026-09-27T01:00:00Z",
            },
        )
        replacement = self.service.create(
            self.coordinator,
            "observation",
            {
                "candidate_id": other_candidate["id"],
                "telescope_id": telescope["id"],
                "team_id": "team-north",
                "mode": "imaging",
                **window,
            },
        )
        replacement = self.service.transition(
            self.coordinator, replacement["id"], "schedule", {"operator_id": "op-1"}
        )
        self.assertEqual(replacement["status"], "scheduled")

    def test_reclassify_keeps_scheduled_observations_pending_review(self):
        candidate, telescope, window, scheduled, requested, completed = (
            self._setup_candidate_with_observations()
        )
        candidate = self.service.transition(
            self.analyst, candidate["id"], "triage", {"reason": "needs typing"}
        )
        result = self.service.transition(
            self.analyst,
            candidate["id"],
            "reclassify",
            {"new_type": "grb", "reason": "spectrum shows broad absorption"},
        )
        self.assertEqual(result["status"], "triaged")
        self.assertEqual(result["data"]["transient_type"], "grb")
        self.assertEqual(result["data"]["previous_type"], "unknown")

        affected = {item["id"]: item for item in result["affected_observations"]}
        self.assertEqual(set(affected), {scheduled["id"]})
        flagged = affected[scheduled["id"]]
        self.assertEqual(flagged["status"], "scheduled")
        self.assertEqual(flagged["data"]["review_status"], "pending")
        # Requested and completed observations are neither changed nor listed.
        self.assertNotIn("review_status", self.service.get(requested["id"])["data"])
        self.assertEqual(self.service.get(completed["id"])["status"], "completed")

        candidate_audit = [
            entry
            for entry in self.service.audit_log(candidate["id"])
            if entry["action"] == "reclassify"
        ]
        self.assertEqual(candidate_audit[0]["detail"]["patch"]["previous_type"], "unknown")
        self.assertEqual(candidate_audit[0]["detail"]["patch"]["transient_type"], "grb")
        flag_audit = [
            entry
            for entry in self.service.audit_log(scheduled["id"])
            if entry["action"] == "flag_review"
        ]
        self.assertEqual(flag_audit[0]["detail"]["previous_type"], "unknown")
        self.assertEqual(flag_audit[0]["detail"]["new_type"], "grb")

    def test_stale_withdraw_updates_nothing(self):
        candidate, telescope, window, scheduled, requested, completed = (
            self._setup_candidate_with_observations()
        )
        audit_before = self.service.audit_log()
        with self.assertRaises(ConflictError):
            self.service.transition(
                self.analyst,
                candidate["id"],
                "withdraw",
                {"reason": "false positive"},
                expected_version=candidate["version"] - 1,
            )
        self.assertEqual(self.service.get(candidate["id"])["status"], "detected")
        self.assertEqual(self.service.get(scheduled["id"])["status"], "scheduled")
        self.assertEqual(self.service.get(requested["id"])["status"], "requested")
        self.assertEqual(
            self.service.get(scheduled["id"])["version"], scheduled["version"]
        )
        self.assertEqual(len(self.service.audit_log()), len(audit_before))

    def test_stale_reclassify_updates_nothing(self):
        candidate, telescope, window, scheduled, requested, completed = (
            self._setup_candidate_with_observations()
        )
        candidate = self.service.transition(
            self.analyst, candidate["id"], "triage", {"reason": "needs typing"}
        )
        audit_before = self.service.audit_log()
        with self.assertRaises(ConflictError):
            self.service.transition(
                self.analyst,
                candidate["id"],
                "reclassify",
                {"new_type": "grb", "reason": "spectrum"},
                expected_version=candidate["version"] - 1,
            )
        self.assertEqual(self.service.get(candidate["id"])["data"]["transient_type"], "unknown")
        self.assertEqual(self.service.get(scheduled["id"])["status"], "scheduled")
        self.assertNotIn("review_status", self.service.get(scheduled["id"])["data"])
        self.assertEqual(len(self.service.audit_log()), len(audit_before))


if __name__ == "__main__":
    unittest.main()
