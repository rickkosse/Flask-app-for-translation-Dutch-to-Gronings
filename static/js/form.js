$("#click-button").on('switch', firstClick)

function firstClick() {
    alert("First Clicked");
    $("#click-button").off('switch').on('click', secondClick)
}

function secondClick() {
    alert("Second Clicked");
    $("#click-button").off('switch').on('click', firstClick)
}

$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				name : $('#nameInput').val()			
			},
			type : 'POST',
			url : '/predict'
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