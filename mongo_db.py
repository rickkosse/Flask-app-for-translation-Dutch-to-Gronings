from creds_mongo import uri
from mongoengine import *
import logging
from bson.objectid import ObjectId

logging.basicConfig(level=logging.INFO)


def connect_to_mongo():
    connect('annotater', host=uri)

    class Gronings_Annotation(Document):
        _id = ObjectIdField()
        orginal_gronings = StringField()
        annotated_gronings = ListField(StringField())
        best_pick = ListField(null=True)
        is_annotated = BooleanField(default=False)

    return Gronings_Annotation


Gronings_Annotation = connect_to_mongo()


def insert_to_db(sentence):
    Gronings_Annotation(orginal_gronings=sentence).save()

    print("Saved to DB")


def get_annotation():
    # instance = Gronings_Annotation.objects.first()
    return Gronings_Annotation.objects.first()


def get_all_annotation():
    return Gronings_Annotation.objects(annotated_gronings__3__exists=False)


def get_all_validations():
    return Gronings_Annotation.objects(annotated_gronings__3__exists=True)


def store_anno_in_mongo(sentence, ids):
    Objected_id = ObjectId(str(ids).strip())
    sentence = str(sentence)
    instance = Gronings_Annotation.objects(_id=Objected_id).get()

    if instance.annotated_gronings:
        print("er betaan al sents")
        Gronings_Annotation.objects(_id=Objected_id).update_one(push__annotated_gronings=sentence)
    else:
        print("er bestaat nog niks")
        sentence=[sentence]
        Gronings_Annotation.objects(_id=Objected_id).update_one(set__annotated_gronings=sentence)


    logging.info("Updated DB")
