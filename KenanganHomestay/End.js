// Initialize Firebase
const firebaseConfig = {
  apiKey: "AIzaSyB8Nv1Nt6RlMiLNj9mab01ixVeUECxPIWY",
  authDomain: "booking-88671.firebaseapp.com",
  databaseURL: "https://booking-88671-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "booking-88671",
  storageBucket: "booking-88671.appspot.com",
  messagingSenderId: "521939658802",
  appId: "1:521939658802:web:8820e7b2514d942195342b",
  measurementId: "G-RY6QSDSMV2"
 };
 
 firebase.initializeApp(firebaseConfig);
 
 // Get a reference to the Firestore service
 const firestore = firebase.firestore();
 
 // Get a reference to the Realtime Database service
 const database = firebase.database();
 
 document.getElementById('bookingForm').addEventListener('submit', function(event) {
  event.preventDefault();
 
  var room = document.getElementById('room').value;
  var date = document.getElementById('date').value;
 
  console.log(`Submitted booking for room ${room} on ${date}`);
 
  // Write the booking to the Firestore database
  firestore.collection('bookings').doc(room).set({
  name: document.getElementById('name').value,
  date: date
  }).then(() => {
  console.log("Booking saved to Firestore!");
  return firestore.collection('bookings').doc(room).get();
  }).then((doc) => {
  console.log("Updated data: ", doc.data());
  }).catch((error) => {
  console.error("Error saving booking to Firestore: ", error);
  });
 
  // Write the booking to the Realtime Database
  database.ref('bookings/' + room + '/' + date).set({
  name: document.getElementById('name').value,
  date: date
  }, (error) => {
  if (error) {
   console.error("Error saving booking to Realtime Database: ", error);
  } else {
   console.log("Booking saved to Realtime Database!");
   return database.ref('bookings/' + room + '/' + date).once('value');
  }
  }).then((snapshot) => {
  console.log("Updated data: ", snapshot.val());
  });
 });
 