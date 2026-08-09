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
    url: "https://resume-worker.pranjalverma0606.workers.dev",
    newTab: true,
  },
];
export const introParagraphs = [
  "I'm Pranjal Verma, a Data Engineer at Accenture in Pune, specializing in building scalable ETL/ELT pipelines, distributed data processing systems, and cloud-native data solutions.",

  "I work primarily with Python, SQL, PySpark, Apache Spark, Airflow, and AWS, with a focus on data pipelines, automation, and reliable data architectures. I also enjoy building open-source projects and writing about data engineering at [https://blog.pvcodes.in]{blog.pvcodes.in}.",
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
        duration: "Sept 2024 - Present",
        current: true,
        points: [
          "Built scalable ETL/ELT pipelines using AWS Glue, PySpark, and Apache Airflow for reliable data processing and analytics.",
          "Optimized distributed Spark workloads on Amazon EMR, increasing pipeline throughput by 3x through partitioning, cluster sizing, and Parquet optimization.",
        ],
      },
    ],
  },

  {
    name: "Walkover",
    url: "https://walkover.in",
    positions: [
      {
        title: "Software Developer Intern",
        duration: "Jan 2024 - Sept 2024",
        current: false,
        points: [
          "Developed backend services and REST APIs using Node.js, Express, PostgreSQL, and MongoDB for production applications.",
          "Implemented asynchronous workflows and optimized database operations to improve backend performance and reliability.",
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
  { text: "Twitter", url: "https://twitter.com/pvcodes" },
  { text: "RSS", url: "/rss.xml" },
];

export const projects = [
  {
    name: "VLR Analytics",
    description:
      "End-to-end cloud data pipeline for scraping and transforming VALORANT esports data using a Medallion Architecture, enabling analytics on player performance, team compositions, and map statistics.",
    url: "https://github.com/pvcodes/vlr-analytics",
    tags: ["ETL Pipeline", "AWS", "PySpark", "Airflow", "Terraform"],
    openSource: true,
  },
  {
    name: "LLMify",
    description:
      "A multi-model LLM chatbot platform supporting different large language models with a unified chat interface.",
    url: "https://llmify.vercel.app",
    tags: ["LLM", "SaaS", "Chatbot", "AI"],
    openSource: false,
  },
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
];
