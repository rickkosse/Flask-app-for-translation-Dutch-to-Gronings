// $("#click-button").on('switch', firstClick)

// function firstClick() {
// 	alert("First Clicked");
// 	$("#click-button").off('switch').on('click', secondClick)
// }

// function secondClick() {
// 	alert("Second Clicked");
// 	$("#click-button").off('switch').on('click', firstClick)
// }
// var state = "nl-gro"
// function state_1() {
// 	if (state == "nl-gro") {
// 		state = "gro-nl";
// 	} else {
// 		state = "nl-gro";
// 	}
// 	document.getElementById("demo").innerHTML = state;
// 	// var myElement = $("demo")= state ; the Jquery variant

// }

var state = "nl-gro"

$(document).ready(function(){
  $(".toggle").click(function(){
    if (state == "nl-gro") {
        state = "gro-nl";
      } else {
        state = "nl-gro";
      }
    // $(".status").html(state);
    console.log(state);
  });
});

$(document).ready(function() {
	$('form').on('submit', function(event) {
		$.ajax({
			data : {
				name : $('#nameInput').val()			
			},
			type : 'POST',
			url : '/predict_'+state
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(data.name).show();
				$('#errorAlert').hide();
			}

		});

		event.preventDefault();

	});

});

// $(document).ready(function() {
// 	$('.toggle').click(function() {
// 		var current_status = $('.status').text();
// 		$.ajax({
// 			url: "/get_toggled_status",
// 			type: "get",
// 			data: {status: current_status},
// 			success: function(response) {
// 				$(".status").html(response);
// 				console.log(state);

// 			},
// 			error: function(xhr) {
//       //Do Something to handle error
//   }
// });
// 	});
// });
