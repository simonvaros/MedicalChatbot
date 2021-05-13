window.chatbotSettings = {
  "botName": "Doctor bot",
  "botImage": 'https://cdn.technologyadvice.com/wp-content/uploads/2018/02/friendly-chatbot.jpg',
  "hubUrl": "http://simonvaros-seq2seq-chatbot.northeurope.azurecontainer.io:5000",
  "autoOpen": true,
  "textInputPlaceholder": "Type your question...",
  "liveChatWelcomeMessage": "What kind of medical issue do you have?"
}

var div = document.createElement('div');
div.id = 'chatbot';
document.getElementsByTagName('body')[0].appendChild(div);

var link = document.createElement('link');
link.rel = 'stylesheet';
link.href = './styles.css';
link.type = 'text/css'
document.getElementsByTagName('head')[0].appendChild(link);

var script = document.createElement('script');
// script.src = '../build/simonvaros-webchat.js';
script.src = 'http://localhost:4000/simonvaros-webchat.js';

document.getElementsByTagName('body')[0].appendChild(script);

var link2 = document.createElement('link');
link2.rel = 'stylesheet';
link2.href = 'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap';
link2.type = 'text/css'
document.getElementsByTagName('head')[0].appendChild(link2);