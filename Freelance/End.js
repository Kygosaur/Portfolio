// Initialize Firebase
const firebaseConfig = {
  apiKey: "AIzaSyB8Nv1Nt6RlMiLNj9mab01ixVeUECxPIWY",
  authDomain: "booking-88671.firebaseapp.com",
  databaseURL: "https://booking-88671-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "booking-88671",
  storageBucket: "booking-88671.appspot.com",
  messagingSenderId: "521939658802",
  appId: "1:521939658802:web:d72bcc5be0184a6895342b",
  measurementId: "G-ZSSER5QHBV"
};

firebase.initializeApp(firebaseConfig);

// Get a reference to the database service
var database = firebase.database();

document.getElementById('bookingForm').addEventListener('submit', function(event) {
  event.preventDefault();

  var room = document.getElementById('room').value;
  var date = document.getElementById('date').value;

  console.log(`Submitted booking for room ${room} on ${date}`);

  // Check if the room is already booked on the selected date
  database.ref('rooms/' + room + '/' + date).once('value').then(function(snapshot) {
    if (snapshot.exists()) {
      // The room is already booked, show an error message
      alert("This room is already booked on " + date);
    } else {
      // The room is not booked, write the booking to the database
      database.ref('rooms/' + room + '/' + date).set({
        name: document.getElementById('name').value,
        date: date
      });
    }
  });
});
