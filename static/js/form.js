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
$(document).ready(function () {
    $('.sent_display').on('click', '.navigate', function () {
        var direction = 'b';
        if ($(this).prop('id') === 'go_forward') {
            direction = 'f';
        }
        $.ajax({
            url: "/get_anno",
            type: "get",
            data: {direction: direction},
            success: function (response) {
                if (response.all.length !== 0) {
                    $('#my-sentence').data("id", response.all[response.count][0]);
                    $('#successAlert').text(response.all[response.count][1]).show();
                    console.log($('#my-sentence').data("id"));
                    if (response.back === "True") {
                        $('.back').html("<button id='go_back' class='navigate btn btn-primary''>Terug</button>")
                    } else {
                        $('#go_back').remove();
                    }
                    console.log(response.forward);
                    if (response.forward === "True") {
                        $('.forward').html("<button id='go_forward' class='navigate btn btn-primary''>Volgende</button>")
                    } else {
                        $('#go_forward').remove();
                    }
                } else {
                    console.log("Empty array");
                    $('#my-sentence').data("id", response.all);
                    $('#successAlert').html("<p id='sent_display'> Helaas! Geen zinnen meer in de database</p>");
                    $('#anno_input').val("");
                    $('#go_forward').prop('disabled', true);
                    $('#sending').prop('disabled', true);


                }
            },
        });
    });
});

$(document).ready(function () {
    $('#help_translating_form').on('submit', function (event) {
        var my_id = $('#my-sentence').data("id");
        $.ajax({
            data: {
                annotation: $('#anno_input').val(),
                original_str: $('#sent_display').text(),
                original_id: my_id.trim(),
            },
            type: 'POST',
            url: '/store_in_mongo',
            success: function (response) {
                console.log("New posted id:", response.count);
                if (response.all.length !== 0) {
                    var index = response.count;
                    $('#my-sentence').data("id", response.all[index][0]);
                    $('#successAlert').text(response.all[index][1]).show();
                    $('#anno_input').val("");
                } else {
                    console.log("Empty array");
                    $('#my-sentence').data("id", response.all);
                    $('#successAlert').html("<p id='sent_display'> Helaas! Geen zinnen meer in de database</p>");
                    $('#anno_input').val("");
                    $('#go_forward').prop('disabled', true);
                    $('#sending').prop('disabled', true);
                }

            },
        });
        event.preventDefault();

    });
});
$(document).ready(function () {
    $('.validation_display').on('click', '.navigate', function () {

        var direction = 'b';
        if ($(this).prop('id') === 'go_forward') {
            direction = 'f';
        }
        $.ajax({
            url: "/get_validations",
            type: "get",
            data: {direction: direction},
            success: function (response) {
                if (response.all.length !== 0) {
                    $('#my-validation').data("id", response.all[response.count][0]);
                    $('#successAlert').text(response.all[response.count][1]).show();
                    if (response.back === "True") {
                        $('.back').html("<button id='go_back' class='navigate btn btn-primary''>Terug</button>")
                    } else {
                        $('#go_back').remove();
                    }
                    console.log(response.forward);
                    if (response.forward === "True") {
                        $('.forward').html("<button id='go_forward' class='navigate btn btn-primary''>Volgende</button>")
                    } else {
                        $('#go_forward').remove();
                    }
                } else {
                    console.log("Empty array");
                    $('#my-validation').data("id", response.all);
                    $('#successAlert').html("<p id='sent_display'> Helaas! Geen zinnen meer in de database</p>");
                    $('#go_forward').prop('disabled', true);
                    $('#sending').prop('disabled', true);
                }
            },
        });
    });
});

$(document).ready(function () {
    $('#help_validation_form').on('submit', function (event) {
        var my_id = $('#my-validation').data("id");
        var checked_array = [];
        var vali = $('#validation_loop').text();
        // console.log($("input[type='checkbox']").val());

        $.each($("input[name='val']:checked"), function () {
            checked_array.push($(this).next().text());
        });
        console.log(checked_array);
        $.ajax({
            data: {
                best_pick: checked_array[0],
                original_id: my_id.trim(),
            },
            type: 'POST',
            url: '/store_validation_in_mongo',
            success: function (response) {
                if (response.all_validations.length !== 0) {
                    console.log("coming through");
                    // console.log(response.data);
                    console.log(response.all_validations);
                    console.log(response.count);
                    var index = response.count - 1;
                    console.log(response.all_validations[response.count]);
                    $('#my-validation').data("id", response.all_validations[response.count][0]);
                    // $('#successAlert').text(response.all_validations[response.count][2]).show();
                    $('#dynamic_appending').remove();
                    $('#dynamic_data').append(response.data);


                } else {
                    console.log("Empty array");
                    $('#my-validation').data("id", response.all_validations);
                    $('#successAlert').html("<p id='sent_display'> Helaas! Geen zinnen meer in de database</p>");
                    $('#go_forward').prop('disabled', true);
                    $('#sending').prop('disabled', true);
                    $('#dynamic_appending').remove();
                    $('#dynamic_data').append(response.data);

                }

            },
        });
        event.preventDefault();

    });
});
$('input.messageCheckbox').on('change', function () {
    $('input.messageCheckbox').not(this).prop('checked', false);
});

var boxes = $('.myCheckBox');

boxes.on('change', function() {
    $('#confirmButton').prop('disabled', !boxes.filter(':checked').length);
}).trigger('change');
