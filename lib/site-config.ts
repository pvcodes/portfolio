export const siteConfig = {
  title: "Pranjal Verma",
  description: "Personal website and blog of Pranjal Verma",
  baseUrl: "https://pvcodes.in",
  author: "Pranjal Verma",
  currentLocation: "Pune, India",
  theme: {
    toggleEnabled: true,
    defaultTheme: "dark" as "light" | "dark",
  },
};

export const navigation = [
  { name: "About", url: "/" },
  { name: "Blog", url: "/blog/" },
  {
    name: "CV",
    url: "https://pvcodes.s3.ap-south-1.amazonaws.com/docs/resume.pdf",
    newTab: true,
  },
];
export const introParagraphs = [
  "I'm Pranjal Verma, a Data Engineer at Accenture in Pune, working on scalable data pipelines, real-time processing, and distributed systems.",
  "I specialize in building production-ready data architectures using Kafka, Spark, Airflow, and BigQuery. I also enjoy automation, cloud-native systems, and writing about data-driven solutions at [https://blog.pvcodes.in]{blog.pvcodes.in}.",
];

export const contactLinks = [
  {
    icon: "/icons/email.svg",
    text: "hello@pvcodes.in",
    url: "mailto:hello@pvcodes.in",
    newTab: false,
  },
  {
    icon: "/icons/email.svg",
    text: "pranjal.b.verma@accenture.com",
    url: "mailto:pranjal.b.verma@accenture.com",
    newTab: false,
  },
  {
    icon: "/icons/twitter.svg",
    text: "@pvcodes",
    url: "https://twitter.com/pvcodes",
    newTab: true,
  },
  {
    icon: "/icons/github.svg",
    text: "pvcodes",
    url: "https://github.com/pvcodes",
    newTab: true,
  },
  {
    icon: "/icons/linkedin.svg",
    text: "pvcodes",
    url: "https://www.linkedin.com/in/pvcodes/",
    newTab: true,
  },
];

export const skills = {
  languages: ["Python", "SQL", "JavaScript", "TypeScript"],
  technologies: [
    "Apache Kafka",
    "Apache Spark",
    "Apache Airflow",
    "BigQuery",
    "Docker",
    "Kubernetes",
    "GCP",
    "AWS",
  ],
};

export const companies = [
  {
    name: "Accenture",
    url: "https://www.accenture.com",
    positions: [
      {
        title: "Data Engineer",
        duration: "2024 - Present",
        current: true,
        points: [
          "Designed and optimized scalable data pipelines for analytics and machine learning workloads.",
          "Worked on real-time data processing using Kafka and Spark.",
          "Built reliable, production-grade data architectures on cloud platforms.",
        ],
      },
    ],
  },
  {
    name: "Walkover",
    url: "https://walkover.in",
    positions: [
      {
        title: "Software Engineer",
        duration: "2022 - 2024",
        current: false,
        points: [
          "Developed microservices and high-performance web applications.",
          "Worked on backend systems supporting large-scale business operations.",
        ],
      },
    ],
  },
];

export const education = [
  {
    degree: "Master of Computer Applications (MCA)",
    field: "Computer Applications",
    institution: "Devi Ahilya Vishwavidyalaya, Indore",
    year: "2024",
    gpa: false,
  },
  {
    degree: "Bachelor of Computer Applications (BCA)",
    field: "Computer Applications",
    institution: "Integral University, Lucknow",
    year: "2022",
    gpa: false,
  },
];
export const footerLinks = [
  { text: "GitHub", url: "https://github.com/pvcodes" },
  { text: "LinkedIn", url: "https://www.linkedin.com/in/pvcodes/" },
  { text: "Twitter", url: "https://twitter.com/undefined810" },
  { text: "RSS", url: "/rss.xml" },
];

export const projects = [
  {
    name: "Kidney Stone Risk Prediction Research",
    description:
      "A predictive research model that identifies kidney stone risk by analyzing individual health factors such as high blood pressure and dietary saturated fatty acid intake.",
    url: "https://github.com/pvcodes/Kidney-Stone-Risk-Prediction-Research",
    tags: ["Machine Learning", "Healthcare", "Research", "Python"],
    openSource: true,
  },
  {
    name: "ERDiagram to JSON",
    description:
      "A fine-tuned Qwen2.5-VL model that converts database ER diagrams into structured JSON schemas, achieving 89.2% table accuracy and 90% relationship accuracy—outperforming the base model.",
    url: "https://github.com/pvcodes/ERDiagram-To-Schema",
    tags: ["LLMs", "Computer Vision", "Databases", "Qwen", "AI"],
    openSource: true,
  },
  {
    name: "LLMify",
    description:
      "A multi-model LLM chatbot platform supporting different large language models with a unified chat interface.",
    url: "https://llmify.xyz",
    tags: ["LLM", "SaaS", "Chatbot", "AI"],
    openSource: false,
  },
];
