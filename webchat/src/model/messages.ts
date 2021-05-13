import { MessageType } from "./messageType";

export class Message {
  type!: MessageType;
}

export class TextMessage extends Message {
  messageText!: string;

  constructor(msg: string) {
    super();
    this.type = MessageType.TextMessage;
    this.messageText = msg;
  }
}

export class UserTextMessage extends Message{
  message: string;

  constructor(msg: string) {
    super();
    this.type = MessageType.UserTextMessage;
    this.message = msg;
  }
}