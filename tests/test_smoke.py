import unittest

from nullcore.orch.app import app, muster, status


class TestSmoke(unittest.TestCase):
    def test_routes_exist(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/muster", paths)
        self.assertIn("/api/status", paths)

    def test_muster_alive(self):
        body = muster()
        self.assertEqual(body["status"], "alive")
        self.assertIsInstance(body["epoch_ms"], int)

    def test_status_scoreboard(self):
        body = status()
        self.assertEqual(body["stage_vector"], [])


if __name__ == "__main__":
    unittest.main()
