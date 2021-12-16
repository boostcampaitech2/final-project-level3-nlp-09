var trial = 0;
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
      var t_element = document.getElementById("trialCount");
      t_element.innerText=trial;
      console.log(trial);
    }
  }
function getHintResponse(trial) {
    if(trial>5){
      document.getElementById("hintButton").disabled = false;
    }
    else{
      document.getElementById("hintButton").disabled = true;
    }
  }
  $("#textInput").keypress(function(e) {
    if (e.keyCode == '13') {
      getBotResponse();
      getHintResponse(trial)
    }
  });
  $("#hintButton").click(function() {
    getHintResponse(trial)
    console.log("Hint");
    console.log(trial);
    trial += 1;
  });