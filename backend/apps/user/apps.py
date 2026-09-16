from django.apps import AppConfig

# to make apps always think about domain not services as many services can be used in multiple domains, but a domain should not be used in multiple services.
class UserConfig(AppConfig):

  default_auto_field = "django.db.models.BigAutoField"
  name = "apps.user"