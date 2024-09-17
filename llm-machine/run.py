import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
with open("./anthropic-api-key.txt", "r") as f:
    anthropic_api_key = f.read().strip()

with open("./openai-api-key.txt", "r") as f:
    openai_api_key = f.read().strip()

os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------------------------------------------------------------------



if __name__ == "__main__":
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an information extraction machine that converts unstructured information into a json format that can be loaded using pythons `json.loads` method and contains every information about: {information}."
            ),
            (
                "assistant",
                "{input}"
            )
        ]
    )
    
    chain = prompt | llm
    
    results = chain.invoke(
        {
            "input": "211 50668 9349253*  * Tel. 0221 9349253*  U ID Nr. : DE 812706034 EUR PERON I 0,0% SX 6.49 A PF AND 0,48 EUR 0.48 A*  TONIC WATER ZERO 3.58 A 2.5 k 1.79 PF AND 0.25 EUR 0.25 0.50 A*  2.5 k 0.25 ORANGE 0. SAFE BRAN G IN CH TE MIX AN AN. DIRE KT S AFT 2.58 A 2.79 SUM ME EUR 21,81 Ge g. EC- Cash EUR 21,81 * *  K unden be leg * *  Datum:  20.03.2024 U hr ze it:  12:56:06 U hr Be leg- Nr.  9741 Trace- Nr.  261921 Kart enz ahl ung Contact less giro card Nr.  56002712 Terminal- ID 00.075.00 Pas- Info AS- Zeit 20.03.  12:56 U hr Be trag EUR 21,81 Zah lung er fol gt Steuer %  Net to Steuer Brutto A= 19,0%  18,33 3,48 21,81 Ges amt be trag 18,33 3,48 21,81 TSE- Signatur:  D Kb Ry 6 PD 7 G 0 pJ p iS RuK/ p 59 fuRt k 7 XB 0 GD ZZ 67 sA 19 BY 0 rF on 080 oG d 3 i 6 T 0 dix G 0 ex 0 s Cs 50 uT i 5 KrB uZ yz Cu 6 Vl Pe 20 ut Hp y 0 Y W j k 4 YN Nb nYZ CD L 1 F Z S ira GU W j By TSE- Signatur zah ler:  3449663 TSE- Trans a ktion:  1675474 TSE- Start:  2024-03-20 T 12:53:39.000 TSE- Stop:  2024-03-20 T 12:54:48.000 Seri en nun mmer Kass e:  RE WE: d 8:5 e: d 3:48:3 f: e 3:00 20.03.2024 12:54 Bon- Nr. : 3089 Mark t: 0446 Kass e: 6 Bed. : 432106 No ch keine PAY BACK Kart e?  Fur dies enE in k auf hate st Du 10 Punk teer hal ten!  G le ichin der RE WE App oder auf www. re we. de/ pay back an mel den.  KeineR abatte oder Punk te auf mit*  gek enz eich net e Prod uk te.  RE WE Markt GmbH",
            "information": "items bought, item count, item price, total price, date, time, receipt number, payment method, market place"
        }
    )
    
    print(results)