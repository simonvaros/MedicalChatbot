declare global {
  interface Window {
    chatbotSettings: BotSettings;
  }
}
export class BotSettings {
  botName!: string;
  botImage!: string;
  hubUrl!: string;
  autoOpen = false;
  liveChatEndpointUrl!: string;
  textInputPlaceholder!: string;
  liveChatWelcomeMessage!: string;
}

export const GlobalSettings: BotSettings = window.chatbotSettings;
