# Generated by Django 3.0.3 on 2020-05-28 11:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face_detector', '0003_testmodel_personid'),
    ]

    operations = [
        migrations.CreateModel(
            name='LastId',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('personId', models.IntegerField(default=0)),
            ],
        ),
    ]