"""GET /api/categories tests (no AI involved)."""

from .helpers import BackendTestCase

EXPECTED = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


class TestCategories(BackendTestCase):
    def test_categories_list(self):
        client, _, _ = self.make_client()
        response = client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"categories": EXPECTED})
