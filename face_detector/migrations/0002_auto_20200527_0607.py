# Generated by Django 3.0.3 on 2020-05-27 03:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face_detector', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='testmodel',
            name='col',
        ),
        migrations.AddField(
            model_name='testmodel',
            name='name',
            field=models.CharField(max_length=40, null=True),
        ),
        migrations.AddField(
            model_name='testmodel',
            name='surname',
            field=models.CharField(max_length=40, null=True),
        ),
    ]