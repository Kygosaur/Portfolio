from google.cloud import firestore
import datetime

def scheduled_function():
   db = firestore.Client()
   current_time = datetime.datetime.utcnow()
   cutoff = current_time - datetime.timedelta(days=2) # 1 week ago

   bookings_ref = db.collection('bookings')
   docs = bookings_ref.where('timestamp', '<', cutoff).stream()

   for doc in docs:
       doc.reference.delete()
