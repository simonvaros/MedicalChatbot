import React from "react";
import { Message } from "../model/messages";
import ChatCanvas from "./ChatCanvas";
import { GlobalSettings } from "../model/botSettings";

type ChatWindowProps = {
  closeButtonClick: () => void;
  messages: Message[];
  botIsTyping: boolean;
  onSendMessage: (message: string, decoder: string) => void;
  showMessageInput: boolean;
};

type ChatWindowState = {
  messageInputText: string;
  decoderSelectValue: string;
};

class ChatWindow extends React.Component<ChatWindowProps, ChatWindowState> {
  messagesEnd: HTMLElement | null;

  constructor(props: ChatWindowProps) {
    super(props);
    this.messagesEnd = null;

    this.state = {
      messageInputText: "",
      decoderSelectValue: "greedy",
    };

    this.handleInputChange = this.handleInputChange.bind(this);
    this.handleInputSubmit = this.handleInputSubmit.bind(this);
    this.onDecoderSelectChange = this.onDecoderSelectChange.bind(this);
  }

  render(): React.ReactNode {
    return (
      <section className="chat">
        <header className="chat-header">
          <div
            className="chat-header-avatar"
            style={{
              backgroundImage: `url(${GlobalSettings.botImage})`,
            }}
          ></div>
          <div className="chat-header-title">{GlobalSettings.botName}</div>
          <div className="chat-header-options">
            <select value={this.state.decoderSelectValue} onChange={this.onDecoderSelectChange}>
              <option value="greedy">Greedy decoder</option>
              <option value="beamsearch">Beam search decoder</option>
              <option value="sampling">Sampling decoder</option>
            </select>
          </div>
          <div className="chat-header-close">
            <button id="close-button" onClick={this.props.closeButtonClick}>
              <div className="chat-close-icon"></div>
            </button>
          </div>
        </header>
        <ChatCanvas
          botIsTyping={this.props.botIsTyping}
          messages={this.props.messages}
        ></ChatCanvas>

        <form
          className="chat-inputarea"
          onSubmit={this.handleInputSubmit}
          hidden={!this.props.showMessageInput}
        >
          <input
            type="text"
            className="chat-input"
            placeholder={GlobalSettings.textInputPlaceholder}
            value={this.state.messageInputText}
            onChange={this.handleInputChange}
          ></input>
          <button type="submit" className="chat-send-btn"></button>
        </form>
      </section>
    );
  }

  handleInputChange(event: any): void {
    this.setState({ messageInputText: event.target.value });
  }

  handleInputSubmit(event: any): void {
    event.preventDefault();
    if (this.state.messageInputText == "") {
      return;
    }

    this.props.onSendMessage(this.state.messageInputText, this.state.decoderSelectValue);
    this.setState({ messageInputText: "" });
  }

  onDecoderSelectChange(event: any): void {
    this.setState({decoderSelectValue: event.target.value});
  }
}

export default ChatWindow;
