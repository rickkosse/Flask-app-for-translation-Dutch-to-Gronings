from creds_mongo import uri
from mongoengine import *
import datetime


def connect_to_mongo():
    connect('annotater', host=uri)

    class Gronings_Annotation(Document):
        _id = IntField()
        orginal_gronings = StringField()
        annotated_gronings = ListField(null=True)
        best_pick = ListField(null=True)

    return Gronings_Annotation


Gronings_Annotation = connect_to_mongo()


def insert_to_db(sentence):
    sentence = Gronings_Annotation(orginal_gronings=sentence).save()

    print("Saved to DB")


def get_annotation():
    instance = Gronings_Annotation.objects.first()
    return [instance.orginal_gronings]


def get_all_annotation():
    return [instance.orginal_gronings for instance in Gronings_Annotation.objects()]
