function getAllIndexes(arr, val) {
    var indexes = [], i = -1;
    while ((i = arr.indexOf(val, i+1)) != -1){
        indexes.push(i);
    }
    return indexes;
}

function intersect(array1,array2) {
    vals = array1.filter(function(n) {
        return array2.indexOf(n) !== -1;
    })
    return vals
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

function sample(n,mx) {
    let randoms = [...Array(n)].map(() => Math.floor(Math.random() * mx));
    return randoms
}

function mean(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average
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


function showStimulus(stim_indx) {
    document.querySelector(`#stimulus_${stim_indx}`).style.display = 'block';
    document.querySelector(`#image_shown`).style.display = 'block';
    listening = true;
}


function feedback(key_pressed) {
    hideAllElements()
    let outcome = reward_probabilities[stim_indx][valid_keys[stim_indx].indexOf(key_pressed)] > Math.random();
    chose_largest = (key_pressed == valid_keys[stim_indx][indexOfMax(reward_probabilities[stim_indx])]);

    if (outcome == 1) {
        document.querySelector(`#feedback_correct`).style.display = 'block';
        document.querySelector(`#rewarded_stim`).style.display = 'block';
    } else {
        document.querySelector(`#feedback_incorrect`).style.display = 'block';
        document.querySelector(`#nonrewarded_stim`).style.display = 'block';
    }

    let reminder = document.querySelector(`#reminder`);
    reminder.innerHTML = `For that image, "${valid_keys[stim_indx][0]}" has a ${reward_probabilities[stim_indx][0]*100}% ` +
        `chance of reward, and "${valid_keys[stim_indx][1]}" has a ${reward_probabilities[stim_indx][1]*100}%` +
        " chance of reward.";
    reminder.style.display = 'block';

    setTimeout( () => {
            runTutorial()
        }
    , 4000);
}


function showInstructions(stimulus_keys, div_indx) {
    hideAllElements()
    document.querySelector(`#instruction_${div_indx}`).style.display = 'block';
    if (stimulus_keys.length > 0) {
        stim_indx = intersect(getAllIndexes(spaceship, stimulus_keys[0]),getAllIndexes(setting, stimulus_keys[1]));
        showStimulus(stim_indx);
        listening = true;
    }
}


function tutorialTask() {
    hideAllElements()
    if (trial_num == 0) {
        document.querySelector(`#instruction_7`).style.display = 'block';
        stimulus_order = sample(n_trials,spaceship.length);
        choice_record = new Array();
        n_sessions++
        setTimeout( () => {
            trial_num++;
            stim_indx = stimulus_order[trial_num-1]
            showStimulus(stim_indx);
        }, 2000)
    } else {
        choice_record.push(chose_largest);
        trial_num++;
        if (trial_num > n_trials) {
            console.log(`Choice record mean ${mean(choice_record)}`)
            if (mean(choice_record) < 0.8) {
                if (n_sessions >= session_max) {
                    document.querySelector(`#goodbye`).style.display = 'block';
                } else {
                    document.querySelector(`#not_good_enough`).style.display = 'block';
                    trial_num = 0;
                    setTimeout( () => {
                        runTutorial();
                    }, 3000)
                }
            } else {
                loop_switch = false;
                runTutorial();
            }
        } else {
            stim_indx = stimulus_order[trial_num-1]
            showStimulus(stim_indx);
        }
    }
}


function runTutorial() {
    hideAllElements()
    switch(instruction_indx) {
        case 0:
            console.log("case 0")
            instruction_indx++;
            loop_switch = false;
            showInstructions(instruction_stimuli[0], 0);
            break;
        case 1:
            console.log("case 1")
            if (! loop_switch) {
                loop_switch = true;
                showInstructions(instruction_stimuli[1], 1);
                break;
            } else {
                if (chose_largest) {
                    loop_switch = false;
                    instruction_indx++;
                } else {
                    showInstructions(instruction_stimuli[1], 1);
                    break;
                }
            }
        case 2:
            console.log("case 2")
            instruction_indx++;
            loop_switch = false;
            showInstructions(instruction_stimuli[2],2);
            break;
        case 3:
            console.log("case 3")
            if (loop_switch){
                if (chose_largest) {
                    loop_switch = false;
                    instruction_indx++;
                } else {
                    showInstructions(instruction_stimuli[3],3);
                    break;
                }
            } else {
                loop_switch = true;
                showInstructions(instruction_stimuli[3],3);
                break;
            }
        case 4:
            instruction_indx++;
            showInstructions(instruction_stimuli[4], 4);
            break;
        case 5:
            instruction_indx++;
            showInstructions(instruction_stimuli[5], 5);
            break;
        case 6:
            loop_switch = true;
            instruction_indx++;
            showInstructions(instruction_stimuli[6], 6);
            break;
        case 7:
            if (loop_switch){
                tutorialTask();
                break;
            } else {
                instruction_indx++;
            }
        case 8:
            document.querySelector(`#instruction_8`).style.display = 'block';
            document.querySelector(`#to-task`).style.display = 'block';
    }

}

document.addEventListener('DOMContentLoaded', function () {
    console.log('in onLoad')
    document.onkeypress = function(event) {
        let key_pressed = event.key;
        if (key_pressed && listening && valid_keys[0].includes(key_pressed)) {
            listening = false
            feedback(key_pressed)
        }
    }
    runTutorial()
});