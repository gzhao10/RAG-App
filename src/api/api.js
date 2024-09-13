
export const getAIMessage = async (userQuery) => {

  /*
    const message = 
    {
      role: "assistant",
      content: "Connect your backend here...."
    }
  */

  try{
    const response = await fetch("http://127.0.0.1:5000/get-ai-message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userQuery }),

    });

    const message = await response.json();
    return message;

    } catch (error){
      return { role: "assistant", content: "Sorry, there was an error processing your request." };
    }
  
};
