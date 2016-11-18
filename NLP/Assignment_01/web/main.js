window.createIME = function (element) {
  var STATUS = {
    FREE: 'free',
    INPUT: 'input',
    LOAD: 'load'
  };

  // IME class
  var IME = function (element) {
    var self = this;
    this.target = element;
    this.createUI();
    this.bindEvents();
    // create timer
    this.timer = this.createShowcandidatesTimer();
    this.delay = 100;

    // clear status
    this.changeToFreeState();

    // candidates settings
    this.candidates = [];
    this.pageCount = 0;
    this.page = 0;
    this.candidatesPerPage = 8;

    // xhr
    this.xhr = new XMLHttpRequest();
    this.endsWithSeparator = false;

    this.xhr.onreadystatechange = function () {
      if (self.xhr.readyState === XMLHttpRequest.DONE) {
        if (self.xhr.status === 200) {
          var json = JSON.parse(self.xhr.responseText);
          self.candidateInput.innerHTML = json.part.join(' ') + (self.endsWithSeparator ? '\'' : '');

          if (json.candidates == null || json.candidates == undefined) {
            json.candidates = [];
          }

          self.loadCandidates(json.candidates);
        } else {
          self.loadCandidates([]);
          console.log("Something Wrong");
        }
      }
    };

    // set focus
    this.target.focus();
  };

  IME.prototype.createUI = function () {
    this.target.className = "ime";
    this.cursorPos = 0;
    this.cursor = document.createElement("span");
    this.cursor.innerHTML = "|";
    this.cursor.className = "ime-selection";
    this.target.appendChild(this.cursor);
    this.target.tabIndex = -1;

    var div = document.createElement("div");
    this.candidateBox = document.createElement("div");
    this.candidateBox.style.display = "none";
    this.candidateBox.className = "ime-candidate";
    this.candidateInput = document.createElement("span");
    this.candidateList = document.createElement("ol");
    document.body.appendChild(this.candidateBox);
    this.candidateBox.appendChild(div);
    div.appendChild(this.candidateInput);
    this.candidateBox.appendChild(this.candidateList);
    this.candidatePagenator = document.createElement("span");
    this.candidatePagenator.className = "ime-pagenator";
    this.candidateBox.appendChild(this.candidatePagenator);
  };

  IME.prototype.createShowcandidatesTimer = function () {
    var self = this;
    return window.setInterval(function () {
      if (self.status === STATUS.INPUT) {
        if (Date.now() - self.lastInputTime > self.delay) {
          self.showcandidates();
        }
      }
    }, 100);
  };

  // event handler binder
  IME.prototype.bindEvents = function () {
    var self = this;
    this.target.addEventListener("keydown", function (e) {
      var key = e.key.toLowerCase();
      e.preventDefault();
      if (self.status === STATUS.FREE) {
        if (key === "backspace") {
          self.removeBefore();
        } else if (key === "delete") {
          if (self.cursorPos < self.target.children.length - 1) {
            self.moveCursor(self.cursorPos + 1);
            self.removeBefore();
          }
        } else if (key === "arrowleft") {
          if (self.cursorPos > 0) {
            self.moveCursor(self.cursorPos - 1);
          }
        } else if (key == "arrowright") {
          if (self.cursorPos < self.target.children.length - 1) {
            self.moveCursor(self.cursorPos + 1);
          }
        } else if (key == "home") {
          self.moveCursor(0);
        } else if (key == "end") {
          self.moveCursor(self.target.children.length - 1);
        } else if (key.length === 1 && key >= 'a' && key <= 'z'){
          self.status = STATUS.INPUT;
          self.showcandidateBox();
          self.candidateInput.innerHTML = key.toLowerCase();
          self.lastInputTime = Date.now();
        } else if (key.length == 1) {
          self.insertCharacter(key);
        } else if (key == "space") {
          self.insertCharacter(" ");
        }
      } else if (self.status === STATUS.INPUT) {
        self.lastInputTime = Date.now();
        if (key == "backspace") {
          self.candidateInput.innerHTML = self.candidateInput.innerHTML.substring(0, self.candidateInput.innerHTML.length - 1);
          if (self.candidateInput.innerHTML === "") {
            self.candidateBox.style.display = "none";
            self.status = STATUS.FREE;
          }
        } else if (key == '+' || key == '-' || key == '=') {
          self.showPage(key == '-' ? self.page - 1 : self.page + 1);
        } else if (key >= '1' && key <= '9') {
          if (self.candidateList.children.length > 0) {
            self.selectcandidate(parseInt(key));
          }
        } else if (key == 'space' || key == ' ') {
            self.selectcandidate(1);
        } else if (key.length === 1) {
          if (key >= 'a' && key <= 'z' || key == '\'') {
            self.candidateInput.innerHTML += key;
          }
        }
      }
    });
  };

  IME.prototype.changeToFreeState = function () {
    this.status = STATUS.FREE;
  };

  // candidate select support
  IME.prototype.showcandidateBox = function () {
    var pos = this.cursor.getClientRects()[0];
    this.candidateBox.style.top = (pos.bottom + 5) + "px";
    this.candidateBox.style.left = (pos.left + 5) + "px";
    this.candidateBox.style.display = "block";
    this.candidateList.innerHTML = "";
  };

  IME.prototype.showcandidates = function (candidates) {
    this.status = STATUS.LOAD;

    if (this.candidateInput.innerHTML[this.candidateInput.innerHTML.length - 1] === '\'') {
        this.endsWithSeparator = true;
    } else {
        this.endsWithSeparator = false;
    }

    this.xhr.open("GET", "/candidates/" + this.candidateInput.innerHTML.replace(/ /g, '\''), true);
    this.xhr.send(null);
    console.log("show");
  };

  IME.prototype.loadCandidates = function (candidates) {
    var html = "", i;
    this.candidates = candidates;
    this.page = 1;
    this.pageCount = Math.ceil(this.candidates.length / this.candidatesPerPage);
    this.showPage(1);
    this.status = STATUS.INPUT;
    this.lastInputTime = NaN;
  };

  IME.prototype.showPage = function (page) {
    var offset, limit;
    if (page >= 1 && page <= this.pageCount) {
      this.page = page;
      page -= 1;
      limit = this.candidatesPerPage;
      offset = page * limit;
      this.showCandidatesList(offset, limit);
      this.candidatePagenator.innerHTML = this.page + "/" + this.pageCount;

      this.lastInputTime = NaN;
    };
  };

  IME.prototype.showCandidatesList = function (offset, limit) {
    var html = "", i;
    if (limit === undefined || limit < 0) {
      limit = 8;
    }

    if (offset === undefined) {
      offset = 0;
    }

    if (offset >= this.candidates.length) {
      offset = this.candidates.length % limit;
      if (offset === 0 && this.candidates.length > 0) {
        offset = limit;
      }
      offset = this.candidates.length - offset;
    }

    for (i = offset; i < this.candidates.length && i < offset + limit; i++) {
      html += "<li>" + this.candidates[i] + "</li>";
    }

    this.candidateList.innerHTML = html;
  };

  IME.prototype.selectcandidate = function (index) {
    var characters, i;
    index -= 1;
    characters = this.candidateList.children[index].innerHTML.split("");
    for (i = 0; i < characters.length; i++) {
      this.insertCharacter(characters[i]);
    }

    this.candidateBox.style.display = "none";
    this.status = STATUS.FREE;
  };

  // input support
  IME.prototype.insertCharacter = function (ch) {
    var span = document.createElement("span");
    span.innerHTML = ch;
    this.target.insertBefore(span, this.cursor);
    this.cursorPos += 1;
    return span;
  };

  IME.prototype.moveCursor = function (newPos) {
    if (newPos < this.cursorPos) {
      this.target.insertBefore(this.cursor, this.target.children[newPos]);
      this.cursorPos = newPos;
    } else if (newPos > this.cursorPos) {
      this.target.insertBefore(this.cursor, this.target.children[newPos + 1]);
      this.cursorPos = newPos;
    }
  };

  IME.prototype.moveRight = function () {
    if (this.cursorPos < this.target.children.length - 1) {
      this.moveCursor(this.cursorPos + 1);
    }
  };

  IME.prototype.moveLeft = function () {
    if (this.cursorPos < this.target.children.length - 1) {
      this.moveCursor(this.cursorPos + 1);
    }
  };

  IME.prototype.moveFront = function () {
    this.moveCursor(0);
  };

  IME.prototype.moveEnd = function () {
    this.moveCursor(this.target.children.length - 1);
  };

  IME.prototype.removeBefore = function () {
    if (this.cursorPos > 0) {
      this.target.children[this.cursorPos - 1].remove();
      this.cursorPos -= 1;
    }
  };

  IME.prototype.removeAfter = function () {
    if (this.cursorPos < this.target.children.length - 1) {
      this.moveCursor(this.cursorPos + 1);
      this.deleteCharacter();
    }
  };

  return function (element) {
    return new IME(element);
  };


} ();

//document.addEventListener("ready", function () {
var input = document.getElementById("input");
var ime = createIME(input);

//});
