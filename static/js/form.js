var state = "nl-gro"
var encoding = "BPE"

$(document).ready(function () {
    $(".toggle").click(function () {
        if (state == "nl-gro") {
            state = "gro-nl";
        } else {
            state = "nl-gro";
        }
        console.log(state);
    });
});


$(document).ready(function () {
    $("#chkToggle2").click(function () {
        if (encoding == "BPE") {
            encoding = "CHAR";
        } else {
            encoding = "BPE";
        }
        console.log(encoding);
    });
});

$(document).ready(function () {
    $('#translate_form').on('submit', function (event) {
        $.ajax({
            data: {
                translation: $('#trans_input').val()
            },
            type: 'POST',
            url: '/predict_' + encoding + '_' + state,
            beforeSend: function () {
                console.log('loading');
                $('#successAlert').hide();
                $('#errorAlert').hide();
                $("#loadingDiv").show();
            },
            success: function (data) {
                console.log('succes');
                $("#loadingDiv").hide();
                $('#successAlert').text(data.translation).show();
                console.log(data.translation);
            }
        })
            .done(function (data) {

                if (data.error) {
                    $('#errorAlert').text(data.error).show();
                    $('#successAlert').hide();
                } else {
                    $('#errorAlert').hide();
                }

            });

        event.preventDefault();

    });

});
$(document).ready(function(){
      $('.sent_display').on('click', '.navigate', function(){
        var direction = 'b';
        if ($(this).prop('id') === 'go_forward'){
          direction = 'f';
        }

        $.ajax({
         url: "/get_anno",
         type: "get",
         data: {direction: direction},
         success: function(response) {
           $('#successAlert').text(response.sentence).show();
           if (response.back === "True"){
             $('.back').html("<button id='go_back' class='navigate btn btn-primary''>Terug</button>")
           }
           else{
             $('#go_back').remove();
           }
           if (response.forward === "True"){
             $('.forward').html("<button id='go_forward' class='navigate btn btn-primary''>Volgende</button>")
           }
           else{
             $('#go_forward').remove();
           }

         },

       });
      });
    });