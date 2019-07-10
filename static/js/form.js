
var state = "nl-gro"
var encoding = "BPE"

$(document).ready(function(){
	$(".toggle").click(function(){
		if (state == "nl-gro") {
			state = "gro-nl";
		} else {
			state = "nl-gro";
		}
		console.log(state);
	});
});


$(document).ready(function(){
	$("#chkToggle2").click(function(){
		if (encoding == "BPE") {
			encoding = "CHAR";
		} else {
			encoding = "BPE";
		}
		console.log(encoding);
	});
});

$(document).ready(function() {
	$('form').on('submit', function(event) {
		$.ajax({
			data : {
				translation : $('#trans_input').val()		
			},
			type : 'POST',
			url : '/predict_'+encoding+'_'+state,
			beforeSend: function() {
				console.log('loading')
				$('#successAlert').hide();
				$('#errorAlert').hide();
				$("#loadingDiv").show();
			},
			success: function(data) {
				console.log('succes')
				$("#loadingDiv").hide();
				$('#successAlert').text(data.translation).show();
			}
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#errorAlert').hide();
			}

		});

		event.preventDefault();

	});

});