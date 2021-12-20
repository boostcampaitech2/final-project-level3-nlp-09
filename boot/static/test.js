var trial = 0;
var correctNum = 0;
var playgame = 1;
var answer = ["거머리", "최우식", "김다미", "ESTP", "스폰지밥"];
var category = ["동물", "연예인", "연예인", "MBTI", "만화주인공"];
// 사용자 프로필 이미지
var userImage =
  '<div class="media media-chat media-chat-reverse"><img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4333/4333609.png" alt="..."><div class="media-body"><p class="userText"><span>';
// 로봇 프로필 이미지
var botImage =
  '<div class="media media-chat"><img class="avatar" src="https://cdn-icons-png.flaticon.com/512/773/773330.png" alt="..."><div class="media-body"><p class="userText"><span>';
// 상태 표시를 위한 flag 저장용 딕셔너리
var dictFlags = {
  selectThema: 0, // 테마 선택 여부
  sendFeedback: -1,
  feedbackMode: 0,
  answerIdx: 0,
};
var saveLogger = {
  context: ["test", "test", "test", "test", "test", "test", "test", "test", "test", "test"], //임의의 context
  userQuestions: [], //사용자가 던진 Y/N 질문 저장
  botAnswers: [0, 1, 0, 1, 0, 1, 0, 1, 1, 1], // Y/N 질문에 대한 로봇의 답 저장 //임의의 답변
  userFeedbackIdx: [], // 사용자가 답함(0 or 1)
};
function getBotResponse() {
  var rawText = $("#textInput").val();
  //console.log(rawText);
  $("#textInput").val("");
  // 시작 flag = selectThema
  if (dictFlags["selectThema"] == 0) {
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    // 시작 버튼을 제대로 입력
    if (rawText == 1) {
      // flag 변경
      dictFlags["selectThema"] = 1;
      // 게임 시작 메시지 출력
      var botStartMessage = `${botImage}****************************************<br>게임을 시작할게!!<br>내가 생각하고 있는 것은 무엇일까?
    ****************************************<br>나는 지금 ${category[dictFlags["answerIdx"]]}중에서 문제를 골랐어!
      <br><b><u>정답:</u></b> 을 앞에 쓰면 정답을 입력할 수 있어!<br>ex) 정답:쿼터백
        </span></p></div></div>`;
      $("#chat-content").append(botStartMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    } else {
      var botHtml = botImage + "게임을 시작하려면 1을 입력해주세요." + "</span></p></div></div>";
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
    //정답을 입력한 경우
  } else if (rawText.indexOf("정답") == 0) {
    var str_len = rawText.length;
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    var botHtml = `${botImage} 정답으로<br>${rawText}를 입력하셨습니다.</span></p></div></div>`;
    $("#chat-content").append(botHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    if (rawText.substring(3, str_len) == answer[dictFlags["answerIdx"]]) {
      dictFlags["answerIdx"] += 1;
      var botAnswerMessage = `${botImage}****************************************<br>정답입니다!!<br>
      ${category[dictFlags["answerIdx"]]}중에서 새로운 문제를 출제했으니 다시 맞춰봐!</span></p></div></div>`;
      $("#chat-content").append(botAnswerMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      correctNum += 1;
      calculateCorrect();
    } else {
      var botAnswerMessage =
        botImage + "****************************************<br>오답입니다!!<br>" + "</span></p></div></div>";
      $("#chat-content").append(botAnswerMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
  } else if (trial < 10) {
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    trial += 1;
    calculateTrial();
    console.log(trial);
    var botHtml = `${botImage} ${trial}번째 질문으로<br>${rawText}를 입력하셨습니다.</span></p></div></div>`;
    $("#chat-content").append(botHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    saveLogger["userQuestions"].push(rawText); //사용자 질문 저장
    if (trial == 10) {
      var botFeedbackMessage = `${botImage} 게임이 종료되었습니다! <br>사용자 피드백을 보내시겠습니까?<br>0: 보내지 않는다. 1: 보낸다.`;
      $("#chat-content").append(botFeedbackMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
  } else if ((trial == 10) & (dictFlags["sendFeedback"] == -1)) {
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    //사용자 피드백 받을지 여부
    if (rawText == 0) {
      // 종료메시지 출력
      var botHtml = `${botImage} 사용자 피드백 보내기 않기를 선택하셨습니다.<br>게임을 종료합니다.`;
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      dictFlags["sendFeedback"] = 0;
    } else if (rawText == 1) {
      //피드백 보내기 실행
      dictFlags["sendFeedback"] = 1;
      dictFlags["feedbackMode"] = 1;
      var botHtml = `${botImage} 사용자 피드백 보내기를 시작합니다.<br>질문에 대한 답이 맞는지 피드백을 보내주세요!`;
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      getUserFeedback();
      // 이하... getuserReedback함수 실행
    } else {
      var botHtml = `${botImage} 잘못 누르셨습니다.<br>0: 보내지 않는다. 1: 보낸다.`;
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
  }
}
function getUserFeedback() {
  // 사용자 입력 출력
  var rawText = $("#textInput").val();
  $("#textInput").val("");

  if (saveLogger["userFeedbackIdx"].length != 0) {
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
  }

  // 질문에 대한 사용자의 답변 저장
  if ((rawText == 0) | (rawText == 1)) {
    saveLogger["userFeedbackIdx"].push(rawText);
  } else {
    var botAlert = `${botImage} 잘못 누르셨습니다.`;
    $("#chat-content").append(botAlert);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
  }

  i = saveLogger["userFeedbackIdx"].length;

  if (i == saveLogger["userQuestions"].length + 1) {
    //종료조건
    //사용자 피드백 파일로 생성
    saveCsv();

    //종료 문구 띄우기
    var botHtml = `${botImage} 사용자 피드백이 전송되었습니다.<br>참여해주셔서 감사합니다!🤗<br>게임을 종료합니다.`;
    $("#chat-content").append(botHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    dictFlags["feedbackMode"] = 2;
  } else {
    // 0 or 1형태 답을 O/X로 사용자에게 보여주기 위함
    if (saveLogger["botAnswers"][i - 1] == 0) {
      var ox = "X";
    } else if (saveLogger["botAnswers"][i - 1] == 1) {
      var ox = "O";
    }
    // 질문할 경우
    var botMessage = `${botImage}
    ***********************************<br>정답: ${answer}<br>
    ***********************************<br>${i}번째 사용자 질문:<br>
    ${saveLogger["userQuestions"][i - 1]}<br>
    ***********************************<br>답변: ${ox}<br>
    ***********************************<br>0: 답변이 틀리다, 1: 답변이 맞다<br>
    ***********************************`;
    $("#chat-content").append(botMessage);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
  }
}

function saveCsv() {
  const data = [["context", "question", "answers", "feedback"]];
  for (var i = 0; i < saveLogger["userQuestions"].length; i++) {
    data.push([
      saveLogger["context"][i],
      saveLogger["userQuestions"][i],
      saveLogger["botAnswers"][i],
      saveLogger["userFeedbackIdx"][i + 1], //맨 처음이 공백이 들어간다..
    ]);
  }

  let csvContent = "data:text/csv;charset=utf-8," + data.map((e) => e.join(",")).join("\n");
  var encodedUri = encodeURI(csvContent);
  var link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "my_data.csv");
  document.body.appendChild(link);
  link.click();
}

function getHintResponse(trial) {
  if (trial >= 5) {
    document.getElementById("hintButton").disabled = false;
  } else {
    document.getElementById("hintButton").disabled = true;
  }
}

function calculateTrial() {
  var t_element = document.getElementById("trialCount");
  t_element.innerText = "전체 질문 횟수 " + trial;
}

function calculateCorrect() {
  var t_element = document.getElementById("correctCount");
  t_element.innerText = "맞힌 갯수 " + correctNum;
}

$("#textInput").keypress(function (e) {
  if ((e.keyCode == "13") & (dictFlags["feedbackMode"] == 0)) {
    getBotResponse();
    getHintResponse(trial);
    calculateTrial();
    calculateCorrect();
  } else if ((e.keyCode == "13") & (dictFlags["feedbackMode"] == 1)) {
    getUserFeedback();
  }
});

$("#hintButton").click(function () {
  getHintResponse(trial);
  console.log("Hint");
  console.log(trial);
  trial = 0;
  calculateTrial();
  getHintResponse(trial);
});
