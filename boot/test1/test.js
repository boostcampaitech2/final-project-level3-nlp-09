function getBotResponse() {
    var rawText = $("#textInput").val();
    console.log(rawText);
    $("#textInput").val("");
    var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
    $("#userchatbox").append(userHtml);
  }
function getHintResponse(trial) {
    if(trial>5){
      document.getElementById("hintButton").disabled = false;
    }
    else{
      document.getElementById("hintButton").disabled = true;
    }
  }
  var trial=0;
  $("#textInput").keypress(function(e) {
    if (e.keyCode == '13') {
      getBotResponse();
      trial+=1;
      getHintResponse(trial)
    }
  });
  $("#hintButton").click(function() {
    getHintResponse(trial)
    console.log("Hint");
    console.log(trial);
    trial += 1;
  });