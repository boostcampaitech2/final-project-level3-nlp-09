var trial = 0;
var correctNum = 0;
function getBotResponse() {
    var rawText = $("#textInput").val();
    console.log(rawText);
    $("#textInput").val("");
    if(rawText >= 1 && rawText <=6){
      var userHtml = '<div class="media media-chat media-chat-reverse"><img class="avatar" src="https://img.icons8.com/color/36/000000/administrator-male.png" alt="..."><div class="media-body"><p class="userText"><span>' + rawText + "</span></p></div></div>";
      $("#chat-content").append(userHtml);
      var botHtml = '<div class="media media-chat"><img class="avatar" src="https://img.icons8.com/color/36/000000/administrator-male.png" alt="..."><div class="media-body"><p class="userText"><span>' + rawText + "번 테마를 골랐습니다." + "</span></p></div></div>";
      $("#chat-content").append(botHtml);
    }
    else{
      var userHtml = '<div class="media media-chat media-chat-reverse"><img class="avatar" src="https://img.icons8.com/color/36/000000/administrator-male.png" alt="..."><div class="media-body"><p class="userText"><span>' + rawText + "</span></p></div></div>";
      $("#chat-content").append(userHtml);
      trial+=1;
      calculateTrial();
      console.log(trial);
    }
  }
function getHintResponse(trial) {
    if(trial>=5){
      document.getElementById("hintButton").disabled = false;
    }
    else{
      document.getElementById("hintButton").disabled = true;
    }
  }
function calculateTrial() {
  var t_element = document.getElementById("trialCount");
  t_element.innerText="전체 질문 횟수 :" + trial;
}
function calculateCorrect() {
  var t_element = document.getElementById("correctCount");
  t_element.innerText="맞힌 갯수 :" + correctNum;
}
  $("#textInput").keypress(function(e) {
    if (e.keyCode == '13') {
      getBotResponse();
      getHintResponse(trial)
      calculateTrial();
      calculateCorrect();
    }
  });
  $("#hintButton").click(function() {
    getHintResponse(trial)
    console.log("Hint");
    console.log(trial);
    trial=0;
    calculateTrial();
    getHintResponse(trial)
  });