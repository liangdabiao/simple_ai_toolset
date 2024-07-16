generate_blog_titles = """I want you to act as a professional blog titles generator. 
Think of titles that are seo optimized and attention-grabbing at the same time,
and will encourage people to click and read the blog post.
They should also be creative and clever.
Try to come up with titles that are unexpected and surprising.
Do not use titles that are too generic,or titles that have been used too many times before. I want to generate 10 titles maximum.
My blog post is is about {topic}
                                 
IMPORTANT: The output should be json array of 10 titles without field names. just the titles! Make Sure the JSON is valid.
                                                  
Example Output:
[
    "Title 1",
    "Title 2",
    "Title 3",
    "Title 4",
    "Title 5",
    "Title 6",
    "Title 7",
    "Title 8",
    "Title 9",
    "Title 10",
]"""


anlysis_prompt = """evaluate the marketing effectiveness of this landing page. Generate a list of 5 suggestions in json, return only the json without anything else.

Output example:
{
  "suggestions": [
    {
      "element": "Headline",
      "suggestion": "Make the headline more specific to the product or service offered to better capture user interest and improve targeting."
    },
    {
      "element": "Value Proposition",
      "suggestion": "Include a more detailed description of benefits and features to clearly communicate the value proposition to visitors."
    },
    {
      "element": "Hero Image",
      "suggestion": "Replace the placeholder with an actual high-quality image or graphic that represents the product or service effectively."
    },
    {
      "element": "Call to Action (CTA)",
      "suggestion": "Make the CTA button more prominent with contrasting colors and actionable text that encourages clicks."
    },
    {
      "element": "Social Proof",
      "suggestion": "Show actual customer testimonials or case studies instead of just a rating to add authenticity and trustworthiness."
    }
  ]
}
"""