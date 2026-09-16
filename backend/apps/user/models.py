# pyrefly: ignore [missing-import]
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
# pyrefly: ignore [missing-import]
from django.db import models


class UserManager(BaseUserManager):

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("Email is required")

        email = self.normalize_email(email)

        user = self.model(
            email=email,
            **extra_fields
        )

        user.set_password(password)
        user.save(using=self._db)

        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True")

        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True")

        return self.create_user(
            email,
            password,
            **extra_fields
        )


class User(AbstractBaseUser, PermissionsMixin):

    email = models.EmailField(unique=True)

    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True)

    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    date_joined = models.DateTimeField(auto_now_add=True)

    objects = UserManager()

    USERNAME_FIELD = "email"

    def __str__(self):
        return self.email


# Create your models here.
'''models.py
     ↓
Python representation of your data
     ↓
Django ORM
     ↓
SQL
     ↓
PostgreSQL
You don't normally write raw SQL for basic database operations.

For example:

User.objects.create(...)

Django's ORM converts your Python operation into SQL that PostgreSQL understands.

So:

models.py describes what your database data looks like.'''
