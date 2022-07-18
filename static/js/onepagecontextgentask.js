function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


function serverUpdate() {
    const csrftoken = getCookie('csrftoken');

    $.ajaxSetup({
        beforeSend: function (xhr, settings) {
            // if not safe, set csrftoken
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

    $.ajax({
        url: updateurl,
        data: {
            // here getdata should be a string so that
            // in your views.py you can fetch the value using get('getdata')
            'responses': JSON.stringify(responses),
            'start_times': JSON.stringify(start_times),
            'end_times': JSON.stringify(end_times)
        },
        dataType: 'json',
        success: function (res, status) {
        },
        error: function (res) {
            // alert(res.status);
        }
    })
    .then(response => {
        hues = JSON.parse(response.obscured);
        stimuli = JSON.parse(response.stimuli);
        last = JSON.parse(response.last);

        if (last) {
            location.href = goodbyeurl;
        }

        block_trial_number = 0
        block_possible = 0;
        block_earned = 0;
        start_times = new Array();
        end_times = new Array();
        responses = new Array();
    })
    .then( () => {
        setTimeout(showStimulus, 2000)
    });

}


function hideAllElements() {
    document.querySelectorAll('div').forEach(div => {
        div.style.display = 'none';
    });
    document.querySelectorAll('h2').forEach(h2 => {
        h2.style.display = 'none';
    });
    document.querySelectorAll('img').forEach(img => {
        img.style.display = 'none';
    });
}

function showStimulus() {
    hideAllElements()
    let stimulus_indx = stimuli[block_trial_number];
    document.querySelector(`#instructions`).style.display = 'block';
    document.querySelector(`#stimulus_${stimulus_indx}`).style.display = 'block';
    document.querySelector(`#image_shown`).style.display = 'block';
    start_times.push(Date.now())
    listening = true;
}


function scrub(dirty){
    const dirty2 = [...dirty]
    let ncolumns = valid_keys.length;
    let nrows = dirty.length / ncolumns;
    let clean = [];
    while(dirty2.length) clean.push(dirty2.splice(0,ncolumns));
    let cntr = 0;
    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncolumns; j++) {
            clean[i][j] = Math.sqrt(atob(clean[i][j])) - cntr;
            cntr++;
        }
    }
    return clean
}


function resultsSummary() {
    hideAllElements()
    let summary = document.querySelector(`#results_summary`);
    summary.innerHTML = 'This block you earned ' + block_earned + ' of ' + block_possible +
        ' possible. In total, you have earned ' + total_earned + ' of ' + total_possible + ' possible.';
    summary.style.display = 'block';

    serverUpdate()

    // setTimeout( function() {
    //     serverUpdate()
    // },
    //     500 // 4000
    // )

    }


function feedback(key_pressed) {
    hideAllElements()

    const clean = scrub(hues)
    let outcome = clean[block_trial_number][valid_keys.indexOf(key_pressed)]

    if (outcome == 1) {
        document.querySelector(`#feedback_correct`).style.display = 'block';
        document.querySelector(`#rewarded_stim`).style.display = 'block';
    } else {
        document.querySelector(`#feedback_incorrect`).style.display = 'block';
        document.querySelector(`#nonrewarded_stim`).style.display = 'block';
    }

    total_possible += Math.max.apply(Math, clean[block_trial_number]);
    block_possible += Math.max.apply(Math, clean[block_trial_number]);
    total_earned += outcome;
    block_earned += outcome;

    setTimeout( () => {
            if (block_trial_number == scrub(hues).length-1) {
                resultsSummary();
                // if (last) {
                //     location.href = goodbyeurl;
                // } else {
                //     resultsSummary();
                // }
            } else {
                block_trial_number++;
                showStimulus();
            }
        }
    , 1000);
}

document.addEventListener('DOMContentLoaded', function () {
    hideAllElements()
    document.querySelector(`#planet_intro`).style.display = 'block';

    setTimeout( () => {
            document.onkeypress = function(event) {
            let key_pressed = event.key;
            if (key_pressed && listening && valid_keys.includes(key_pressed)) {
                listening = false
                end_times.push(Date.now())
                responses.push(key_pressed)
                feedback(key_pressed)
            }
        }
        showStimulus()
    }, 10000);


});

