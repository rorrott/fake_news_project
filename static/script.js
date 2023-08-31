$(document).ready(function() {
    $('#news-form').submit(function(event) {
        event.preventDefault();
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: $(this).serialize(),
            dataType: 'json', // Add this line to specify JSON response
            success: function(response) {
                console.log(response);
                $('#prediction').html('<div class="prediction-container"><h4>Prediction:</h4><p>' + response.prediction + '</p></div>');
            },
            error: function() {
                $('#prediction').html('<h4>Prediction:</h4><p>An error occurred.</p>');
            }
        });
    });
});

