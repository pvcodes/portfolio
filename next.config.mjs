/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  async redirects() {
    return [
      {
        source: '/cv',
        destination: '/resume.pdf',
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
