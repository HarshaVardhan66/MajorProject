from django.db import models


class Susinfo(models.Model):
    name = models.CharField(max_length=100)
    susid = models.CharField(max_length=20)
    auth = models.CharField(max_length=100)
    phone = models.CharField(max_length=20)
    isSuspect = models.CharField(max_length=10)

    def __str__(self):
        return self.name + '-' + self.susid



