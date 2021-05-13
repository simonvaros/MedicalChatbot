import React from "react";
import {
  Message,
  TextMessage,
  UserTextMessage,
} from "../model/messages";
import { MessageType } from "../model/messageType";
import TextMsg from "./TextMsg";
import TypingIndicator from "./TypingIndicator";

type ChatCanvasProps = {
  messages: Message[];
  botIsTyping: boolean;
};

class ChatCanvas extends React.Component<ChatCanvasProps> {
  messagesEnd: HTMLElement | null;

  constructor(props: ChatCanvasProps) {
    super(props);
    this.messagesEnd = null;
  }

  scrollToBottom(): void {
    if (this.messagesEnd != null) {
      this.messagesEnd.scrollIntoView({ behavior: "smooth" });
    }
  }

  componentDidMount(): void {
    this.scrollToBottom();
  }

  componentDidUpdate(): void {
    this.scrollToBottom();
  }

  render(): React.ReactNode {
    const lastMsgIndex = this.props.messages.length - 1;
    return (
      <main className="chat-canvas">
        {this.props.messages.map((message, i) => {
          let showAvatar = i == lastMsgIndex;
          if (this.props.botIsTyping) {
            showAvatar = false;
          }

          switch (message.type) {
            case MessageType.TextMessage:
              const txtMsg = message as TextMessage;
              return (
                <TextMsg
                  isFromBot={true}
                  messageText={txtMsg.messageText}
                  showAvatar={showAvatar}
                ></TextMsg>
              );
            case MessageType.UserTextMessage:
              const usrTxtMessage = message as UserTextMessage;
              return (
                <TextMsg
                  isFromBot={false}
                  messageText={usrTxtMessage.message}
                  showAvatar={showAvatar}
                ></TextMsg>
              );
          }
        })}
        {this.props.botIsTyping && <TypingIndicator></TypingIndicator>}
        <div
          style={{ float: "left", clear: "both" }}
          ref={(el) => {
            this.messagesEnd = el;
          }}
        ></div>
      </main>
    );
  }
}

export default ChatCanvas;
