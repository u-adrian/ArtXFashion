# Generated by Django 3.2.13 on 2022-06-18 09:35

from django.db import migrations, models
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='testfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.CharField(max_length=30)),
                ('file_image', models.ImageField(upload_to='images')),
            ],
        ),
        migrations.CreateModel(
            name='Transfer',
            fields=[
                ('token', models.CharField(default=uuid.uuid4, editable=False, max_length=50, primary_key=True, serialize=False)),
                ('person_image', models.ImageField(upload_to='images/randomseed')),
                ('style_image', models.ImageField(upload_to='images/randomseed')),
                ('person_image_segmentation', models.ImageField(default='loading.gif', upload_to='images/')),
                ('style_transfered', models.ImageField(default='loading.gif', upload_to='images/')),
                ('created_date', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('segment_start_x', models.IntegerField(default=0)),
                ('segment_start_y', models.IntegerField(default=0)),
            ],
        ),
    ]
