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
        source = self.service.create(
            self.analyst, "source", {"name": "Survey", "survey_name": "S"}
        )
        self.candidate = self.service.create(
            self.analyst,
            "candidate",
            {
                "source_id": source["id"],
                "event_id": "AT-cascade",
                "ra": 10,
                "dec": 20,
                "magnitude": 18,
                "transient_type": "unknown",
                "observed_at": "2026-09-27T00:00:00Z",
            },
        )
        self.telescope = self.service.create(
            self.coordinator,
            "telescope",
            {"name": "North 2m", "aperture_m": 2.0, "site_name": "NAO"},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _observation(self, start="2026-09-28T10:00:00Z", end="2026-09-28T11:00:00Z"):
        return self.service.create(
            self.coordinator,
            "observation",
            {
                "candidate_id": self.candidate["id"],
                "telescope_id": self.telescope["id"],
                "team_id": "team-north",
                "start_at": start,
                "end_at": end,
                "mode": "imaging",
            },
        )

    def _triage(self):
        self.service.transition(
            self.analyst, self.candidate["id"], "triage", {"reason": "reviewed"}
        )

    def test_withdraw_cancels_pending_observations_and_releases_windows(self):
        scheduled = self._observation()
        scheduled = self.service.transition(
            self.coordinator, scheduled["id"], "schedule", {"operator_id": "op-1"}
        )
        requested = self._observation(
            start="2026-09-28T12:00:00Z", end="2026-09-28T13:00:00Z"
        )

        updated = self.service.transition(
            self.analyst,
            self.candidate["id"],
            "withdraw",
            {"reason": "bogus detection"},
        )
        self.assertEqual(updated["status"], "withdrawn")
        cancelled = {obs["id"]: obs for obs in updated["cancelled_observations"]}
        self.assertEqual(set(cancelled), {scheduled["id"], requested["id"]})

        for obs_id in (scheduled["id"], requested["id"]):
            obs = self.service.get(obs_id)
            self.assertEqual(obs["status"], "withdrawn")
            self.assertEqual(obs["data"]["withdrawn_reason"], "bogus detection")
            self.assertEqual(obs["data"]["withdrawn_by"], "analyst-1")
            entries = self.service.audit_log(obs_id)
            withdraw_entries = [e for e in entries if e["action"] == "withdraw"]
            self.assertEqual(len(withdraw_entries), 1)
            self.assertEqual(withdraw_entries[0]["actor_id"], "analyst-1")
            self.assertEqual(withdraw_entries[0]["detail"]["reason"], "bogus detection")

        # The telescope and team windows are free again for the same slot.
        follow_up = self._observation()
        follow_up = self.service.transition(
            self.coordinator, follow_up["id"], "schedule", {"operator_id": "op-2"}
        )
        self.assertEqual(follow_up["status"], "scheduled")

    def test_reclassify_flags_scheduled_observations_for_review(self):
        self._triage()
        scheduled = self._observation()
        scheduled = self.service.transition(
            self.coordinator, scheduled["id"], "schedule", {"operator_id": "op-1"}
        )

        updated = self.service.transition(
            self.analyst,
            self.candidate["id"],
            "reclassify",
            {"new_type": "grb", "reason": "spectra match"},
        )
        self.assertEqual(updated["data"]["transient_type"], "grb")
        self.assertEqual(updated["data"]["previous_type"], "unknown")
        flagged = updated["flagged_observations"]
        self.assertEqual([obs["id"] for obs in flagged], [scheduled["id"]])

        obs = self.service.get(scheduled["id"])
        self.assertEqual(obs["status"], "scheduled")
        self.assertTrue(obs["data"]["review_pending"])

        candidate_audit = self.service.audit_log(self.candidate["id"])
        reclassify = [e for e in candidate_audit if e["action"] == "reclassify"][0]
        self.assertEqual(reclassify["detail"]["patch"]["previous_type"], "unknown")
        self.assertEqual(reclassify["detail"]["patch"]["transient_type"], "grb")
        obs_audit = self.service.audit_log(scheduled["id"])
        flag = [e for e in obs_audit if e["action"] == "flag_review"][0]
        self.assertEqual(flag["detail"]["previous_type"], "unknown")
        self.assertEqual(flag["detail"]["new_type"], "grb")

        # The scheduled observation still holds the telescope window.
        clash = self._observation()
        with self.assertRaises(ConflictError):
            self.service.transition(
                self.coordinator, clash["id"], "schedule", {"operator_id": "op-2"}
            )

    def test_stale_version_leaves_all_entities_untouched(self):
        self._triage()
        scheduled = self._observation()
        scheduled = self.service.transition(
            self.coordinator, scheduled["id"], "schedule", {"operator_id": "op-1"}
        )

        with self.assertRaises(ConflictError):
            self.service.transition(
                self.analyst,
                self.candidate["id"],
                "withdraw",
                {"reason": "stale"},
                expected_version=999,
            )
        candidate = self.service.get(self.candidate["id"])
        self.assertEqual(candidate["status"], "triaged")
        obs = self.service.get(scheduled["id"])
        self.assertEqual(obs["status"], "scheduled")
        self.assertNotIn("withdrawn_reason", obs["data"])

        with self.assertRaises(ConflictError):
            self.service.transition(
                self.analyst,
                self.candidate["id"],
                "reclassify",
                {"new_type": "grb", "reason": "stale"},
                expected_version=999,
            )
        candidate = self.service.get(self.candidate["id"])
        self.assertEqual(candidate["data"]["transient_type"], "unknown")
        obs = self.service.get(scheduled["id"])
        self.assertNotIn("review_pending", obs["data"])


if __name__ == "__main__":
    unittest.main()
