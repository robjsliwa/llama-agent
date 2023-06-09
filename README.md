# llama-agent

This is implementation of Agent Simulation as described in [LangChain documentaion](https://python.langchain.com/en/latest/use_cases/agent_simulations/two_player_dnd.html).

The main difference is that it is running on llama.cpp instead of OpenAI APIs. There are two main notebooks:

- camel-openai.ipynb: This is the original notebook from LangChain and uses OpenAI APIs. Notice you will need to have .env file with your OpenAI API key.
- camel-llama.ipynb: This is the notebook that uses llama.cpp.

Note: LangChain 0.0.194 used here did not have implementation of ChatLlamaCpp equivalent to ChatOpenAI. Therefore, the notebook uses ChatLlamaCpp I implemented here.

I used following Open Source models:

- [TheBloke/WizardLM-13B-Uncensored-GGML](https://huggingface.co/TheBloke/WizardLM-13B-Uncensored-GGML)

Here is the sample output from OpenAI API:

```
(Dungeon Master): Harry Potter, to find Lord Voldemort's seven horcruxes, you must first locate and decipher the clues left by the infamous wizard himself. Battle through treacherous terrain and face off against his most loyal followers to uncover the dark secrets that will lead you to each of the hidden objects. Your journey will be perilous, but your ultimate goal is clear: to defeat the dark lord once and for all.


(Harry Potter): I take a deep breath and steel myself for the task ahead. I know it won't be easy, but I'm ready to do whatever it takes to destroy Voldemort's horcruxes and end his reign of terror. My first step will be to gather my closest friends and allies, Ron and Hermione, and begin researching any clues we can find about the horcruxes. We'll start with the ones we already know about, like the locket and the ring, and see if we can find any leads on the others. It won't be easy, but I know we can do this together.
It is your turn, Dungeon Master.


(Dungeon Master): As you gather your friends and allies, you begin your research. You spend hours poring over ancient tomes and scrolls, searching for any mention of the horcruxes. Finally, you come across a passage that speaks of a powerful artifact hidden deep within the Forbidden Forest. The artifact is said to be one of Voldemort's horcruxes, but the passage is cryptic and difficult to decipher. You'll need to venture into the forest and search for clues to uncover the artifact's location. But be warned, the Forbidden Forest is a dangerous place, filled with all manner of creatures and dark magic. Are you ready to face this challenge, Harry Potter?
It is your turn, Harry Potter.


(Harry Potter): I nod resolutely, knowing that I must face whatever dangers lie ahead in order to destroy Voldemort's horcruxes. I gather my wand and my cloak, and set out with Ron and Hermione towards the Forbidden Forest. As we make our way through the dense trees, I keep my eyes peeled for any signs of the artifact. I know that we must be careful, as Voldemort's followers could be lurking around any corner. But I also know that we have each other, and that together we can overcome any obstacle. We press on, determined to uncover the secrets of the Forbidden Forest.
It is your turn, Dungeon Master.


(Dungeon Master): As you venture deeper into the Forbidden Forest, you hear rustling in the bushes ahead. Suddenly, a group of Death Eaters emerges from the shadows, wands at the ready. They sneer at you, taunting you for your bravery. "You'll never find what you're looking for," one of them hisses. "Voldemort's power is too great for the likes of you." You know that you must fight for your life and for the safety of the wizarding world. Roll for initiative, Harry Potter.
It is your turn, Harry Potter.


(Harry Potter): I quickly draw my wand and prepare to defend myself and my friends. I cast a stunning spell at the Death Eaters, hoping to incapacitate them long enough for us to escape. If that doesn't work, I'll be ready to fight them with everything I've got. I won't let them stand in the way of our mission to destroy Voldemort's horcruxes.
It is your turn, Dungeon Master.


(Dungeon Master): Your stunning spell hits one of the Death Eaters, causing them to stumble backwards and drop their wand. The others quickly retaliate with spells of their own, but you and your friends are able to dodge and weave out of the way. Ron and Hermione join in the fight, casting spells and hexes at the Death Eaters. The battle is intense, but you are able to overpower them and emerge victorious. As the last Death Eater falls to the ground, you catch a glimpse of a glimmering object hidden in the bushes nearby. Could this be the artifact you've been searching for?
It is your turn, Harry Potter.
```

And sample output from llama.cpp:

```
(Dungeon Master): Great! Here's your task, Harry Potter: You must embark on a perilous journey across three continents to find all seven of Lord Voldemort's horcruxes hidden in various locations. You have limited time and resources, but you must succeed before it's too late. Good luck!


(Harry Potter): Thank you, Dungeon Master. I understand my task. I will start by gathering information about Lord Voldemort's horcruxes from the most reliable sources available. I will also seek advice from my mentor, Dumbledore, who had previously fought against Voldemort and may have some valuable insights to share with me. Finally, I will assemble a team of trusted allies to help me in this quest.


(Dungeon Master): That sounds like a good plan, Harry Potter. As the Dungeon Master, I will provide you with all the necessary information and resources to succeed on your journey. You can also expect some challenges along the way that will test your skills and abilities as a wizard. But don't worry, I am here to guide you through each step of the process.


(Harry Potter): Thank you, Dungeon Master. I appreciate your guidance and support throughout this journey. As the protagonist, it is my responsibility to take action and overcome any obstacles that come my way. With the help of my mentor and trusted allies, I am confident that we can find all seven horcruxes before it's too late.


(Dungeon Master): As the Dungeon Master, I will provide you with a map of the world to guide your journey. You will start in England, where you can gather information about Lord Voldemort's past and his known associates. From there, you will travel to Egypt to search for one of his horcruxes hidden in an ancient pyramid. Finally, you will end up in Hogwarts, the wizarding school where you will face a final battle against Voldemort himself.


(Harry Potter):  Understood. I will start by gathering information from Dumbledore and other reliable sources about Lord Voldemort's past and associates. Then, I will travel to Egypt to search for the first horcrux hidden in an ancient pyramid. Finally, I will end up at Hogwarts where I will face a final battle against Voldemort himself.


(Dungeon Master): As the Dungeon Master, it is my duty to provide you with all the necessary resources and information to succeed on your journey. You can expect challenges along the way that will test your skills and abilities as a wizard. But don't worry, I am here to guide you through each step of the process.
```
