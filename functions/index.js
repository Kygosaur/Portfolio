const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();

exports.scheduledFunction = functions.pubsub.schedule('0 9 * * *').timeZone('Asia/Singapore').onRun((context) => {
 const currentTime = admin.firestore.Timestamp.now();
 const cutoff = currentTime.seconds - (2 * 24 * 60 * 60); // 2 days ago

 return admin.firestore().collection('bookings')
   .where('timestamp', '<', cutoff)
   .get()
   .then(snapshot => {
     snapshot.docs.forEach(doc => {
       doc.ref.delete();
     });
   });
});
