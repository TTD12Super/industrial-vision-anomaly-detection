"""History / detail / delete API tests (MOCK inference, fake history).

Inference is mocked (``MockDetector``); history uses the REAL
``DetectionRepository`` code over an in-memory fake collection. No MongoDB
server or DINOv2 model is needed.
"""

from .helpers import BackendTestCase, make_png_bytes


def _detect(client, category="bottle"):
    return client.post(
        "/api/detect",
        data={"category": category},
        files={"file": ("img.png", make_png_bytes(), "image/png")},
    )


class TestHistoryList(BackendTestCase):
    def _seed(self, client, count=3, category="bottle"):
        ids = []
        for _ in range(count):
            response = _detect(client, category)
            self.assertEqual(response.status_code, 200)
            ids.append(response.json()["prediction_id"])
        return ids

    def test_post_persists_history(self):
        client, _, _ = self.make_client()
        prediction_id = _detect(client).json()["prediction_id"]
        self.assertTrue(self.repo.exists(prediction_id))

    def test_list_pagination(self):
        client, _, _ = self.make_client()
        self._seed(client, count=3)
        body = client.get("/api/detections?page=1&page_size=2").json()
        self.assertEqual(body["total"], 3)
        self.assertEqual(len(body["items"]), 2)
        self.assertEqual(body["page"], 1)
        self.assertEqual(body["page_size"], 2)
        second = client.get("/api/detections?page=2&page_size=2").json()
        self.assertEqual(len(second["items"]), 1)
        for item in body["items"]:
            self.assertNotIn("_id", item)
            self.assertIn("input_url", item["result"])

    def test_category_filter(self):
        client, _, _ = self.make_client()
        self._seed(client, count=2, category="bottle")
        self._seed(client, count=1, category="cable")
        body = client.get("/api/detections?category=cable").json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["category"], "cable")

    def test_invalid_category_filter_returns_400(self):
        client, _, _ = self.make_client()
        response = client.get("/api/detections?category=phone")
        self.assertEqual(response.status_code, 400)

    def test_score_filter(self):
        client, _, _ = self.make_client()
        self._seed(client, count=2)  # mock score is always 0.42
        inside = client.get("/api/detections?min_score=0.1&max_score=0.5").json()
        self.assertEqual(inside["total"], 2)
        outside = client.get("/api/detections?min_score=0.5").json()
        self.assertEqual(outside["total"], 0)

    def test_invalid_score_range_returns_400(self):
        client, _, _ = self.make_client()
        response = client.get("/api/detections?min_score=0.9&max_score=0.1")
        self.assertEqual(response.status_code, 400)

    def test_invalid_sort_order_returns_400(self):
        client, _, _ = self.make_client()
        response = client.get("/api/detections?sort_order=random")
        self.assertEqual(response.status_code, 400)


class TestDetectionDetail(BackendTestCase):
    def test_detail_ok(self):
        client, _, _ = self.make_client()
        prediction_id = _detect(client).json()["prediction_id"]
        response = client.get(f"/api/detections/{prediction_id}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prediction_id"], prediction_id)
        self.assertEqual(body["category"], "bottle")
        self.assertIn("created_at", body)
        self.assertNotIn("_id", body)
        self.assertNotIn("artifacts", body)

    def test_detail_missing_returns_404(self):
        client, _, _ = self.make_client()
        response = client.get("/api/detections/" + "f" * 32)
        self.assertEqual(response.status_code, 404)


class TestDeleteDetection(BackendTestCase):
    def test_delete_flow(self):
        client, storage, _ = self.make_client()
        prediction_id = _detect(client).json()["prediction_id"]
        response = client.delete(f"/api/detections/{prediction_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Detection deleted successfully", "prediction_id": prediction_id},
        )
        self.assertFalse(self.repo.exists(prediction_id))
        self.assertFalse(storage.prediction_dir(prediction_id).exists())
        self.assertEqual(client.get(f"/api/detections/{prediction_id}").status_code, 404)

    def test_delete_missing_returns_404(self):
        client, _, _ = self.make_client()
        response = client.delete("/api/detections/" + "f" * 32)
        self.assertEqual(response.status_code, 404)
