from creds_mongo import uri
from mongoengine import *
import logging
from bson.objectid import ObjectId
import os
logging.basicConfig(level=logging.INFO)

base_value = 500

uri = os.environ.get("URI")
def connect_to_mongo():
    connect('annotater', host=uri)

    class Gronings_Annotation(Document):
        _id = ObjectIdField()
        orginal_gronings = StringField()
        annotated_gronings = ListField(StringField())
        best_pick = ListField()
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
    return Gronings_Annotation.objects(annotated_gronings__2__exists=False)[:base_value]


def get_all_validations():
    return Gronings_Annotation.objects(Q(annotated_gronings__2__exists=True) and Q(best_pick__1__exist=False))


def store_anno_in_mongo(sentence, ids):
    Objected_id = ObjectId(str(ids).strip())
    sentence = str(sentence).strip()
    instance = Gronings_Annotation.objects(_id=Objected_id).get()
    if instance.annotated_gronings:
        Gronings_Annotation.objects(_id=Objected_id).update_one(add_to_set__annotated_gronings=sentence)
    else:
        sentence = [sentence]
        Gronings_Annotation.objects(_id=Objected_id).update_one(set__annotated_gronings=sentence)

    logging.info("Updated DB")


def store_valid_in_mongo(best_pick, ids):
    Objected_id = ObjectId(str(ids).strip())
    best_pick = str(best_pick)
    if best_pick == "Geen van allen":
        # Gronings_Annotation.objects(_id=Objected_id).update(pull_all__annotated_gronings=[])
        instance = Gronings_Annotation.objects(_id=Objected_id).get()
        removable = instance.annotated_gronings
        Gronings_Annotation.objects(_id=Objected_id).update(pull_all__annotated_gronings=removable)

        logging.info("removed from val DB")

    else:
        Gronings_Annotation.objects(_id=Objected_id).update_one(best_pick=best_pick)

        logging.info("Updated DB for validation")


def replete_anno_db(read_items):
    return Gronings_Annotation.objects(Q(annotated_gronings__2__exists=False) and Q(_id__nin=read_items))[
           :base_value]


def replete_valid_db(read_vals):
    return Gronings_Annotation.objects(
        Q(annotated_gronings__2__exists=True) & Q(best_pick__size=0) & Q(_id__nin=read_vals))[:base_value]
