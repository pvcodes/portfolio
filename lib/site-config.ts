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
        duration: "Sept 2024 - Present",
        current: true,
        points: [
          "Optimized BigQuery data warehouse on GCP with advanced partitioning and clustering — cutting query latency by 60% and operational costs by 35%.",
          "Built real-time event-driven pipelines using Cloud Run, EventArc, and Apache Kafka to automate business workflows at scale.",
          "Orchestrated ETL workflows with Apache Airflow delivering curated datasets to Elasticsearch, MySQL, PostgreSQL, and Kafka for live analytics.",
        ],
      },
    ],
  },
  {
    name: "Walkover",
    url: "https://walkover.in",
    positions: [
      {
        title: "Data Engineer",
        duration: "Jan 2024 - Sept 2024",
        current: false,
        points: [
          "Designed backend data infrastructure for a workflow automation platform in TypeScript and PostgreSQL, handling 10,000+ concurrent users.",
          "Architected a fault-tolerant pipeline with RabbitMQ message queuing, boosting system throughput by 50% via async processing.",
          "Improved data retrieval performance by 30% through batched access patterns and lazy-loading strategies across high-traffic database operations.",
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
      "End-to-end data pipeline scraping VLR.gg to surface advanced VALORANT esports metrics — player performance trends, team compositions, map win rates, and more. Built for analysts and betting researchers who need signal beyond the box score.",
    url: "https://github.com/pvcodes/vlr-analytics",
    tags: ["ETL Pipeline", "Airflow", "Terraform", "Google Cloud", "Pyspark"],
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
