import React from "react";
import BotButton from "./components/BotButton";
import { GlobalSettings } from "./model/botSettings";
import { Message, UserTextMessage, TextMessage } from "./model/messages";
import ChatWindow from "./components/ChatWindow";
import { MessageType } from "./model/messageType";

type AppProps = {
  isChatOpen: boolean;
};

type AppState = {
  isChatOpen: boolean;
  messages: Message[];
  botIsTyping: boolean;
  showMessageInput: boolean;
};
class App extends React.Component<AppProps, AppState> {
  constructor(props: AppProps) {
    super(props);

    let chatOpen = localStorage.getItem("preduChatOpen");
    if (chatOpen == null) {
      localStorage.setItem("preduChatOpen", GlobalSettings.autoOpen.toString());
      chatOpen = GlobalSettings.autoOpen.toString();
    }

    if (
      window.matchMedia("screen and (max-width: 640px)").matches ||
      window.matchMedia("screen and (max-height: 640px)").matches
    ) {
      chatOpen = "false";
    }

    this.state = {
      isChatOpen: chatOpen == "true",
      messages: [],
      botIsTyping: false,
      showMessageInput: true,
    };
    this.botButtonClick = this.botButtonClick.bind(this);
    this.closeButtonClick = this.closeButtonClick.bind(this);
    this.onSendMessage = this.onSendMessage.bind(this);
  }

  botButtonClick(): void {
    localStorage.setItem("preduChatOpen", "true");

    this.setState({ isChatOpen: true });
  }

  async onSendMessage(message: string, decoder: string): Promise<void> {
    const userMessage = new UserTextMessage(message);

    this.setState({ messages: this.state.messages.concat(userMessage) });

    const url = GlobalSettings.hubUrl + "?question=" + message + "&decoder=" + decoder;

    const response = await fetch(url);
    const result = await response.text();

    const botMessage = new TextMessage(result);
    this.setState({ messages: this.state.messages.concat(botMessage) });
  }

  closeButtonClick(): void {
    localStorage.setItem("preduChatOpen", "false");
    this.setState({ isChatOpen: false });
  }

  render(): React.ReactNode {
    return (
      <div>
        {this.state.isChatOpen ? (
          <ChatWindow
            closeButtonClick={this.closeButtonClick}
            messages={this.state.messages}
            botIsTyping={this.state.botIsTyping}
            onSendMessage={this.onSendMessage}
            showMessageInput={this.state.showMessageInput}
          ></ChatWindow>
        ) : (
          <BotButton botButtonClick={this.botButtonClick}></BotButton>
        )}
      </div>
    );
  }
}

export default App;
