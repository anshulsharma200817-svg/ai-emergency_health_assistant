# pyrefly: ignore [missing-import]
from django.test import TestCase
# pyrefly: ignore [missing-import]
from django.urls import reverse
# pyrefly: ignore [missing-import]
from rest_framework import status
# pyrefly: ignore [missing-import]
from rest_framework.test import APIClient
# pyrefly: ignore [missing-import]
from .models import User
import io
import sys


class UserRegistrationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = reverse("register")
        self.valid_payload = {
            "email": "testuser@example.com",
            "password": "strongpassword123",
            "first_name": "Test",
            "last_name": "User",
        }

    def test_register_user_success(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            response = self.client.post(self.register_url, self.valid_payload, format="json")
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(email="testuser@example.com").exists())
        user = User.objects.get(email="testuser@example.com")
        self.assertEqual(user.first_name, "Test")
        self.assertEqual(user.last_name, "User")
        
        # Verify terminal output outputted the success message
        output = captured_output.getvalue()
        self.assertIn("[REGISTER SUCCESS] New User Registered!", output)
        self.assertIn("testuser@example.com", output)

    def test_register_duplicate_email(self):
        User.objects.create_user(
            email="testuser@example.com",
            password="existingpassword",
        )

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            response = self.client.post(self.register_url, self.valid_payload, format="json")
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        output = captured_output.getvalue()
        self.assertIn("[REGISTER FAILED]", output)
        self.assertIn("A user with this email already exists", response.content.decode())

    def test_register_short_password(self):
        invalid_payload = self.valid_payload.copy()
        invalid_payload["password"] = "123"

        response = self.client.post(self.register_url, invalid_payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("password", response.json())

