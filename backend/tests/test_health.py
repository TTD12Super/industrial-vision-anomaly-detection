"""GET /api/health tests (no AI involved; database state is stubbed)."""

from .helpers import BackendTestCase


class TestHealth(BackendTestCase):
    def test_health_connected(self):
        client, _, _ = self.make_client(db_status="connected")
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "industrial-vision-anomaly-detection",
                "database": "connected",
            },
        )

    def test_health_degraded(self):
        client, _, _ = self.make_client(db_status="disconnected")
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "status": "degraded",
                "service": "industrial-vision-anomaly-detection",
                "database": "disconnected",
            },
        )
