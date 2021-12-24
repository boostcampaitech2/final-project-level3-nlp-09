var totaltrial = 0; //전체 질문 횟수
var correctNum = 0; //맞춘 문제 개수
var playgame = 1; //게임을 시작한지에 대한 flag
var problemtrial = 0; //한문제당 질문한 횟수

var answer_list = [
  "ESFJ",
  "ESFP",
  "ESTJ",
  "ESTP",
  "INFP",
  "INTP",
  "헤르메스",
  "헤파이스토스",
  "똘똘이 스머프",
  "파파 스머프",
  "농구",
  "미식축구",
  "배구",
  "배드민턴",
  "스쿼시",
  "야구",
  "축구",
  "태권도",
  "테니스",
  "펜싱",
  "금성",
  "목성",
  "수성",
  "지구",
  "천왕성",
  "토성",
  "해왕성",
  "화성",
]; //정답
var category_list = ["mbti", "그리스로마신화(등장인물)", "스머프", "운동", "행성"]; //카테고리
var answer = ""; //정답
var category = ""; //카테고리
// 사용자 프로필 이미지
var userImage =
  '<div class="media media-chat media-chat-reverse"><img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4333/4333609.png" alt="..."><div class="media-body"><p class="userText"><span>'; //유저 이미지
// 로봇 프로필 이미지
var botImage =
  '<div class="media media-chat"><img class="avatar" src="https://cdn-icons-png.flaticon.com/512/773/773330.png" alt="..."><div class="media-body"><p class="userText"><span>'; //봇 이미지
// 상태 표시를 위한 flag 저장용 딕셔너리
var dictFlags = {
  selectThema: 0, // 테마 선택 여부
  sendFeedback: -1,
  feedbackMode: 0,
  answerIdx: 0,
};
var saveLogger = {
  answer_keyword: [], //정답 저장
  userQuestions: [], //사용자가 던진 Y/N 질문 저장
  botAnswers: [], // Y/N 질문에 대한 로봇의 답 저장 //임의의 답변
  userFeedbackIdx: [], // 사용자가 답함(0 or 1)
};
//엔터를 쳤을 때 일어나는 일
function getBotResponse() {
  //사용자 입력을 가져옴
  var rawText = $("#textInput").val();
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
      var tmp = pickQuestion(); //답과 카테고리 정함
      answer = tmp[0]; //답
      category = tmp[1]; //카테고리
      console.log(answer);
      console.log(category);

      /* TODO: category, answer를 api에게 전달*/
      $.get("/set_category", { category: category, answer: answer }).done(function (data) {
        // data: BoolQA model의 response, Extractived-based MRC model의 출력
        console.log(data);
        console.log("send category and answer!!");
      });
      $.get("/set_category_boolq", { category: category, answer: answer }).done(function (data) {
        // data: BoolQA model의 response, Extractived-based MRC model의 출력
        console.log(data);
        console.log("send category and answer!!");
      });

      var botStartMessage = `${botImage}****************************************<br>게임을 시작할게!!!<br>내가 생각하고 있는 것은 무엇일까?
    ****************************************<br>나는 지금 <b><u> ${category}</u></b> 카테고리에서 문제를 골랐어!
      <br><b><u>정답:</u></b> 을 앞에 쓰면 정답을 입력할 수 있어!<br>ex) 정답:쿼터백
        </span></p></div></div>`;
      $("#chat-content").append(botStartMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    } else if (rawText == 2) {
      //게임 방법 설명 칸
      var botStartMessage = `${botImage}입력 칸에 질문을 입력하고<br> 엔터를 누르면 O,X 질문을 할 수 있어!<br>한 문제에서 5번 이상 틀리면 Hint를 쓸 수 있어!<br>
        Hint는 입력 칸에 주관식 질문을 쓰고 Hint 버튼을 눌러!<br>정답을 입력할 때는 <b><u>정답:떡볶이</u></b> 이런 형태로 입력해!
        </span></p></div></div>`;
      $("#chat-content").append(botStartMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      var botMes = `${botImage}<span>ㅎㅇ! 나는 Q100의 QBot이야.<br /></span>
                      <span>한 문제에서 5번 이상 틀리면 Hint를 사용할 수 있어!<br /></span>
                      <span>게임 시작을 원하면 1을 입력해!</span><br />
                      <span>게임 방법이 궁금하면 2를 입력해!</span></p></div></div>`;
      $("#chat-content").append(botMes);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    } else {
      //게임 시작을 올바르게 하지 않은 경우
      var botHtml = botImage + "게임을 시작하려면 1을 입력해주세요." + "</span></p></div></div>";
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
    //정답을 입력한 경우
  } else if (rawText.indexOf("정답") == 0) {
    var str_len = rawText.length;
    // 유저가 입력한 정답 값 보여주기
    var userHtml = userImage + rawText + "</span></p></div></div>";
    $("#chat-content").append(userHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);

    //정답인 경우
    console.log(`정답=${answer}, 입력=${rawText.substring(3, str_len)}`);
    if (rawText.substring(3, str_len) == answer) {
      var tmp = pickQuestion(); //답과 카테고리 정함
      answer = tmp[0]; //답
      category = tmp[1]; //카테고리
      console.log(answer);
      console.log(category);
      var botAnswerMessage = `${botImage}****************************************<br>정답입니다!!<br>
      <b><u>${category}</u></b> 카테고리에서 새로운 문제를 출제했으니 다시 맞춰봐!</span></p></div></div>`;
      $("#chat-content").append(botAnswerMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      correctNum += 1;
      problemtrial = 0;
      calculateCorrect();
      totaltrial += 1;
      calculateTrial();
    } else {
      //오답인 경우
      var botAnswerMessage =
        botImage + "****************************************<br>오답입니다!!<br>" + "</span></p></div></div>";
      $("#chat-content").append(botAnswerMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      totaltrial += 1;
      problemtrial += 1;
      calculateTrial();
    }
    if (totaltrial == 20) {
      var botFeedbackMessage = `${botImage} 게임이 종료되었습니다! <br>사용자 피드백을 보내시겠습니까?<br>0: 보내지 않는다. 1: 보낸다.`;
      $("#chat-content").append(botFeedbackMessage);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
    }
  } else if (totaltrial < 20) {
    // boolq 질문을 하는 경우
    //사용자 질문 보여주기
    $.get("/get_boolq", { msg: rawText }).done(function (data) {
      console.log(data);
      // data: BoolQA model의 response, Extractived-based MRC model의 출력
      var userHtml = userImage + rawText + "</span></p></div></div>";
      $("#chat-content").append(userHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      totaltrial += 1;
      problemtrial += 1;
      calculateTrial();
      console.log(totaltrial);
      //모델 정답 출력
      var botAnswer = `${botImage} ${rawText}에 대한 답은<br>${data} 입니다.</span></p></div></div>`;
      $("#chat-content").append(botAnswer);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);

      // 잘 모르겠는 경우 빼고 저장(yes or no)
      if ((changeBoolans(data) == 1) | (changeBoolans(data) == 0)) {
        saveLogger["userQuestions"].push(rawText); //사용자 질문 저장
        saveLogger["botAnswers"].push(data); //모델 답 저장
        saveLogger["answer_keyword"].push(answer); // 실제 답 저장
      }

      if (totaltrial == 20) {
        var botFeedbackMessage = `${botImage} 게임이 종료되었습니다! <br>사용자 피드백을 보내시겠습니까?<br>0: 보내지 않는다. 1: 보낸다.`;
        $("#chat-content").append(botFeedbackMessage);
        $("#chat-content")
          .stop()
          .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      }
    });
  } else if ((totaltrial == 20) & (dictFlags["sendFeedback"] == -1)) {
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
    if (changeBoolans(saveLogger["botAnswers"][i - 1]) == 0) {
      var ox = "X";
    } else if (changeBoolans(saveLogger["botAnswers"][i - 1]) == 1) {
      var ox = "O";
    }
    // 질문할 경우
    var botMessage = `${botImage}
    ***********************************<br>정답: ${saveLogger["answer_keyword"][i]}<br>
    ***********************************<br>${i}번째 사용자 O/X질문:<br>
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
  const data = [["answer_keyword", "question", "answers", "feedback"]];
  for (var i = 0; i < saveLogger["userQuestions"].length; i++) {
    data.push([
      saveLogger["answer_keyword"][i],
      saveLogger["userQuestions"][i],
      changeBoolans(saveLogger["botAnswers"][i]),
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
// 네 맞습니다, 아니오. 틀립니다를 1,0으로 변경하는 함수
function changeBoolans(data) {
  if (data == "네 맞습니다.") {
    return 1;
  } else if (data == "아니오. 틀립니다.") {
    return 0;
  } else if (data == "잘 모르겠습니다.") {
    return -1;
  } else {
    return -2;
  }
}
//random으로 문제와 카테고리 return 해주는 함수
function pickQuestion() {
  var randomQuestionIdx = Math.floor(Math.random() * 28);
  if (randomQuestionIdx >= 0 && randomQuestionIdx <= 5) {
    return [answer_list[randomQuestionIdx], category_list[0]];
  } else if (randomQuestionIdx >= 6 && randomQuestionIdx <= 7) {
    return [answer_list[randomQuestionIdx], category_list[1]];
  } else if (randomQuestionIdx >= 8 && randomQuestionIdx <= 9) {
    return [answer_list[randomQuestionIdx], category_list[2]];
  } else if (randomQuestionIdx >= 10 && randomQuestionIdx <= 19) {
    return [answer_list[randomQuestionIdx], category_list[3]];
  } else {
    return [answer_list[randomQuestionIdx], category_list[4]];
  }
}

//한 문제당 질문 횟수가 5이상이면 힌트 버튼 활성화
function getHintResponse(problemtrial) {
  if (problemtrial >= 5) {
    document.getElementById("hintButton").disabled = false;
  } else {
    document.getElementById("hintButton").disabled = true;
  }
}
//전체 질문 횟수를 업데이트 해주는 함수
function calculateTrial() {
  var t_element = document.getElementById("trialCount");
  t_element.innerText = "전체 질문 횟수 " + totaltrial;
}
//맞춘 문제 개수를 업데이트 해주는 함수
function calculateCorrect() {
  var t_element = document.getElementById("correctCount");
  t_element.innerText = "맞힌 갯수 " + correctNum;
}
//엔터를 입력한 경우 boolq 모델
$("#textInput").keypress(function (e) {
  if ((e.keyCode == "13") & (dictFlags["feedbackMode"] == 0)) {
    getBotResponse();
    getHintResponse(problemtrial);
    calculateTrial();
    calculateCorrect();
  } else if ((e.keyCode == "13") & (dictFlags["feedbackMode"] == 1)) {
    getUserFeedback();
  }
});
//힌트를 클릭하는 경우 주관식 모델
$("#hintButton").click(function () {
  getHintResponse(problemtrial);
  var rawText = $("#textInput").val();
  $("#textInput").val("");
  console.log(rawText);
  //질문이 없는데 힌트 버튼을 누른경우
  if (rawText.length == 0) {
    var botHtml = `${botImage} 주관식 질문을 입력하세요.<br>`;
    $("#chat-content").append(botHtml);
    $("#chat-content")
      .stop()
      .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
  } else {
    //질문이 있으면서 힌트 버튼을 누른경우 주관식 모델
    $.get("/get_hint", { msg: rawText }).done(function (data) {
      console.log(data);
      // data: BoolQA model의 response, Extractived-based MRC model의 출력
      var userHtml = userImage + rawText + "</span></p></div></div>";
      $("#chat-content").append(userHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      var botHtml = `${botImage} Hint 질문으로<br>${rawText}를 입력하셨습니다.</span></p></div></div>`;
      $("#chat-content").append(botHtml);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      var botAnswer = `${botImage} ${rawText}에 대한 답은<br>${data} 입니다.</span></p></div></div>`;
      $("#chat-content").append(botAnswer);
      $("#chat-content")
        .stop()
        .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      totaltrial += 1;
      problemtrial += 1;
      calculateTrial();
      getHintResponse(problemtrial);
      if (totaltrial == 20) {
        var botFeedbackMessage = `${botImage} 게임이 종료되었습니다! <br>사용자 피드백을 보내시겠습니까?<br>0: 보내지 않는다. 1: 보낸다.`;
        $("#chat-content").append(botFeedbackMessage);
        $("#chat-content")
          .stop()
          .animate({ scrollTop: $("#chat-content")[0].scrollHeight }, 1000);
      }
    });
  }
});
