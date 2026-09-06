"""
Integration Unit Tests for SIH_ML REST API Server Endpoints (Section 19 Frontend Readiness)
"""

import unittest
import json
import urllib.request
import urllib.error
import threading
import time
import sys
from pathlib import Path

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api import HealthcareMLRequestHandler, HTTPServer


class TestAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 8089
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server = HTTPServer(("127.0.0.1", cls.port), HealthcareMLRequestHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_01_health_check(self):
        req = urllib.request.Request(f"{self.base_url}/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["ayush_parameters"], 10)
            self.assertTrue(data["agni_koshtha_excluded"])

    def test_02_full_intake_and_summary_api_workflow(self):
        # 1. Create Session
        payload = json.dumps({"patient_id": "PAT_API_001", "language_code": "hi-IN"}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/intake/session", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            session = json.loads(resp.read().decode("utf-8"))
            session_id = session["session_id"]
            self.assertEqual(session["language"]["code"], "hi-IN")

        # 2. Patient Message
        payload = json.dumps({"session_id": session_id, "text": "पेट में जलन और एसिडिटी"}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/intake/message", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res_msg = json.loads(resp.read().decode("utf-8"))
            self.assertIn("status", res_msg)

        # 3. Document Attachment
        payload = json.dumps({"session_id": session_id, "file_path": "report.pdf"}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/intake/document", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            doc_res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(doc_res["file_name"], "report.pdf")

        # 4. Generate Summary
        payload = json.dumps({"session_id": session_id}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/summary/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            summary = json.loads(resp.read().decode("utf-8"))
            summary_id = summary["summary_id"]
            self.assertEqual(summary["status"], "AI_GENERATED_DRAFT")

        # 5. Physician Edit (PUT)
        payload = json.dumps({"physician_edits": {"chief_complaint": "Acute hyperacidity"}, "physician_notes": "Verified"}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/summary/{summary_id}", data=payload, headers={"Content-Type": "application/json"}, method="PUT")
        with urllib.request.urlopen(req) as resp:
            edited = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(edited["status"], "PHYSICIAN_EDITED")

        # 6. Physician Approve
        payload = json.dumps({"physician_id": "DR_API_TEST", "physician_signature": "Dr. Test"}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/summary/{summary_id}/approve", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            approved = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(approved["status"], "FINALIZED_APPROVED")
            self.assertIn("input_sha256_hash", approved["governance_audit"])

        # 7. GET Summary Documents
        req = urllib.request.Request(f"{self.base_url}/summary/{summary_id}/documents")
        with urllib.request.urlopen(req) as resp:
            docs = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(docs["documents_count"], 1)


if __name__ == "__main__":
    unittest.main()
