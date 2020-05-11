from creds_mongo import uri
from mongoengine import *
import logging


logging.basicConfig(level=logging.INFO)


def connect_to_mongo():
    connect('annotater', host=uri)

    class Gronings_Annotation(Document):
        _id = ObjectIdField()
        orginal_gronings = StringField()
        annotated_gronings = ListField(StringField(null=True))
        best_pick = ListField(null=True)
        is_annotated = BooleanField(default=False)

    return Gronings_Annotation


Gronings_Annotation = connect_to_mongo()


def insert_to_db(sentence):
    sentence = Gronings_Annotation(orginal_gronings=sentence).save()

    print("Saved to DB")


def get_annotation():
    # instance = Gronings_Annotation.objects.first()
    return Gronings_Annotation.objects.first()


def get_all_annotation():
    return Gronings_Annotation.objects(annotated_gronings__3__exists=False)


def get_all_validations():
    return Gronings_Annotation.objects(Q(annotated_gronings__ne="") and Q(best_pick=None))


def store_anno_in_mongo(sentence, ids):
    Gronings_Annotation.objects(_id=ids).update(annotated_gronings=sentence)
    logging.info("Updated DB")
