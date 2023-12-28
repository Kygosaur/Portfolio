const express = require('express');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.urlencoded({ extended: true }));

app.post('/booking', (req, res) => {
 const { name, room, date } = req.body;

 // Here you would save the data to your database
 // For example, if you were using MongoDB, you might do something like this:
 // db.collection('bookings').insertOne({ name, room, date })

 res.send('Booking successful!');
});

app.listen(3000, () => console.log('Server running on port 3000'));
