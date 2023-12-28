$(document).ready(function(){
    $('#bookingForm').on('submit', function(e){
      e.preventDefault();
   
      var name = $('#name').val();
      var room = $('#room').val();
      var date = $('#date').val();
   
      $.ajax({
        url: 'your_server_url', // replace with your server URL
        type: 'post',
        data: {name: name, room: room, date: date},
        success: function(response){
          alert("Booking successful!");
        },
        error: function(error){
          console.log(error);
        }
      });
    });
   });
   