from django.db import models
from django.contrib.auth.models import AbstractUser


class MwmblUser(AbstractUser):
    pass


class Curation(models.Model):
    user = models.ForeignKey(MwmblUser, on_delete=models.CASCADE, null=True)
    timestamp = models.DateTimeField()
    query = models.CharField(max_length=300)

    # The original results as stored in the index
    original_index_results = models.JSONField(default=list)

    # The original results that the user saw. May include results from Google via the extension.
    original_results = models.JSONField()
    new_results = models.JSONField()
    num_changes = models.IntegerField(default=0)
