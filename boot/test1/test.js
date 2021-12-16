var trial = 0;
var correctNum = 0;
function getBotResponse() {
    var rawText = $("#textInput").val();
    console.log(rawText);
    $("#textInput").val("");
    var userHtml = '<div class="media media-chat media-chat-reverse"><img class="avatar" src="https://img.icons8.com/color/36/000000/administrator-male.png" alt="..."><div class="media-body"><p class="userText"><span>' + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    trial += 1;
    calculateTrial();
    console.log(trial);
  }
function getHintResponse(trial) {
  /*
  엔터가 눌릴 때마다, hint 버튼을 활성화시킬지 결정
  */
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

count = 0

$("#textInput").keypress(function(e) {
  if (e.keyCode == '13') {
    count += 1
    if (count >  10){
      if ($("#textInput").val() == "재시작") {
        count = 0;
        window.location.reload();
      }
      return;
    }
    // .scrollIntoView({block: "start", behavior: "smooth"});
    $("#chat-content").scrollTop($("#chat-content").height());
    console.log($("#chat-content").height())
    //   $("html, chat-content").animate({
      //     scrollTop: $(// 
        //       'html, chat-content').get(0).scrollHeight 
        // }
        // }, 2000);
    getBotResponse();
    getHintResponse(trial)
    calculateTrial();
    calculateCorrect();
    document.getElementById("userInput")
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